from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from meddiagnosis.xai.visualization import AttributionResult, overlay_heatmap


class LocalVLM(Protocol):
    model: torch.nn.Module

    def prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[dict[str, Any], int]: ...

    def get_vision_module(self) -> torch.nn.Module: ...


def _encoder_layers(vision: torch.nn.Module):
    if hasattr(vision, "encoder") and hasattr(vision.encoder, "layers"):
        return vision.encoder.layers
    inner = getattr(vision, "vision_model", None)
    if inner is not None and hasattr(getattr(inner, "encoder", None), "layers"):
        return inner.encoder.layers
    return None


def _find_target_layer(vision: torch.nn.Module, layer_index: int = -1) -> torch.nn.Module:
    """Pick a vision encoder block.

    The final block tends to produce diffuse, class-agnostic maps; intermediate
    blocks localise better (see Grad-CAM-for-ViT literature), so layer_index is
    configurable rather than hardcoded to the last layer.
    """
    layers = _encoder_layers(vision)
    if layers is not None and len(layers) > 0:
        idx = layer_index if -len(layers) <= layer_index < len(layers) else -1
        return layers[idx]
    for module in reversed(list(vision.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return vision


def _compute_cam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """Grad-CAM from matched activation/gradient tensors."""
    if acts.shape != grads.shape:
        min_c = min(acts.shape[-1], grads.shape[-1])
        acts = acts[..., :min_c]
        grads = grads[..., :min_c]

    if acts.ndim == 4:
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
    elif acts.ndim == 3:
        weights = grads.mean(dim=1, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
    else:
        cam = (acts * grads).mean(dim=-1)

    return F.relu(cam)


def _tokens_to_grid(cam_1d: np.ndarray) -> np.ndarray:
    """Reshape token-level CAM to a square grid when possible."""
    if cam_1d.ndim != 1:
        return cam_1d
    n = cam_1d.size
    side = int(round(n**0.5))
    if side * side == n:
        return cam_1d.reshape(side, side)
    # Padding a non-square token count invents spatial structure that isn't in
    # the model, so refuse instead of fabricating a plausible-looking map.
    raise RuntimeError(
        f"Token CAM of length {n} is not a square grid; cannot map it to image space."
    )


def _target_score(
    logits: torch.Tensor,
    tokenizer: Any | None,
    answer: str | None,
) -> torch.Tensor:
    """Score to differentiate for attribution.

    `logits.max()` is "how confident is the model about anything", which barely
    varies with the question -- it yields near-identical maps for unrelated
    prompts. Differentiating the logit of a specific answer token instead makes
    the attribution answer-specific, which is the point of a class activation map.
    """
    row = logits[0, 0]
    if answer and tokenizer is not None:
        ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(ids) == 1 and 0 <= ids[0] < row.shape[-1]:
            return row[ids[0]]
    return row[row.argmax()]


def _downscale_for_attribution(image: Image.Image, max_size: int | None) -> Image.Image:
    if not max_size or max(image.size) <= max_size:
        return image
    scale = max_size / max(image.size)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.BILINEAR)


@contextmanager
def _single_crop(model_wrapper: LocalVLM):
    """Disable image tiling during attribution.

    SmolVLM splits an image into tiles plus a global view. The resulting token
    sequence is several independent grids concatenated, so reshaping it into one
    square interleaves crops and produces a brick-pattern artifact rather than a
    spatial map. One crop means one grid that can be reshaped honestly.
    """
    proc = getattr(model_wrapper, "processor", None)
    targets = [o for o in (proc, getattr(proc, "image_processor", None)) if o is not None]
    saved = [(o, o.do_image_splitting) for o in targets if hasattr(o, "do_image_splitting")]
    for obj, _ in saved:
        obj.do_image_splitting = False
    try:
        yield bool(saved)
    finally:
        for obj, prev in saved:
            obj.do_image_splitting = prev


@contextmanager
def _attribution_context(model_wrapper: LocalVLM, device: str | None):
    """Run attribution on `device` in float32, then restore the model.

    MPS float16 backward passes crash the Metal shader compiler on large VLM
    graphs, taking the host down with them. CPU/float32 is slower but survives.
    """
    if not device:
        yield None
        return

    model = model_wrapper.model
    original_device = next(model.parameters()).device
    original_dtype = next(model.parameters()).dtype
    target = torch.device(device)
    if original_device == target and original_dtype == torch.float32:
        yield torch.float32
        return

    model.to(target, dtype=torch.float32)
    try:
        yield torch.float32
    finally:
        model.to(original_device, dtype=original_dtype)


def _input_gradient_saliency(
    model_wrapper: LocalVLM,
    image: Image.Image,
    prompt: str,
    input_len: int,
    inputs: dict[str, Any],
    alpha: float,
    answer: str | None = None,
) -> AttributionResult:
    """Fallback saliency map from pixel gradients."""
    pixel_key = next(
        (k for k in ("pixel_values", "images") if k in inputs and torch.is_tensor(inputs[k])),
        None,
    )
    if pixel_key is None:
        raise RuntimeError("No pixel tensor found for saliency fallback.")

    pixels = inputs[pixel_key].detach().clone().requires_grad_(True)
    step_inputs = dict(inputs)
    step_inputs[pixel_key] = pixels

    model_wrapper.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model_wrapper.model(**step_inputs)
        logits = outputs.logits[:, input_len - 1 : input_len, :]
        proc = getattr(model_wrapper, "processor", None)
        score = _target_score(logits, getattr(proc, "tokenizer", None), answer)
        score.backward()

    grad = pixels.grad
    if grad is None:
        raise RuntimeError("Pixel gradients were not computed.")
    attr = grad.detach().abs()
    while attr.ndim > 2:
        attr = attr.mean(dim=0)
    attr = attr.float().cpu().numpy()
    overlay = overlay_heatmap(image, attr, alpha=alpha)
    return AttributionResult(method="input_gradient", heatmap=attr, overlay=overlay)


class GradCAMExplainer:
    """Grad-CAM over the vision encoder for VLM faithfulness analysis."""

    def __init__(
        self,
        model: LocalVLM,
        prefer_fast: bool = False,
        max_image_size: int | None = 512,
        attribution_device: str | None = "cpu",
        layer_index: int = -1,
    ) -> None:
        self.model_wrapper = model
        self.prefer_fast = prefer_fast
        self.max_image_size = max_image_size
        self.attribution_device = attribution_device
        self.layer_index = layer_index
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self, layer: torch.nn.Module) -> None:
        def forward_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            self.gradients = grad.detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def explain(
        self,
        image: Image.Image,
        prompt: str,
        alpha: float = 0.45,
        answer: str | None = None,
    ) -> AttributionResult:
        small = _downscale_for_attribution(image, self.max_image_size)
        if small.size != image.size:
            print(
                f"XAI: {image.width}x{image.height} -> {small.width}x{small.height} "
                "to keep the backward graph small.",
                file=sys.stderr,
            )

        with _attribution_context(self.model_wrapper, self.attribution_device), _single_crop(
            self.model_wrapper
        ):
            if self.prefer_fast:
                print("Using input-gradient saliency (xai.fast).", file=sys.stderr)
                inputs, input_len = self.model_wrapper.prepare_inputs(small, prompt)
                return _input_gradient_saliency(
                    self.model_wrapper, image, prompt, input_len, inputs, alpha, answer
                )

            vision = self.model_wrapper.get_vision_module()
            layer = _find_target_layer(vision, self.layer_index)
            self._register_hooks(layer)

            try:
                inputs, input_len = self.model_wrapper.prepare_inputs(small, prompt)
                self.model_wrapper.model.zero_grad(set_to_none=True)

                with torch.enable_grad():
                    outputs = self.model_wrapper.model(**inputs)
                    logits = outputs.logits[:, input_len - 1 : input_len, :]
                    proc = getattr(self.model_wrapper, "processor", None)
                    score = _target_score(logits, getattr(proc, "tokenizer", None), answer)
                    score.backward()

                if self.activations is None or self.gradients is None:
                    raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

                cam = _compute_cam(self.activations.float(), self.gradients.float())
                cam_np = cam[0].detach().cpu().numpy()
                if cam_np.ndim == 1:
                    cam_np = _tokens_to_grid(cam_np)
                elif cam_np.ndim > 2:
                    cam_np = cam_np.mean(axis=0)

                overlay = overlay_heatmap(image, cam_np, alpha=alpha)
                return AttributionResult(method="gradcam", heatmap=cam_np, overlay=overlay)
            except Exception as exc:
                print(f"Grad-CAM failed ({exc}); using input-gradient fallback.", file=sys.stderr)
                inputs, input_len = self.model_wrapper.prepare_inputs(small, prompt)
                return _input_gradient_saliency(
                    self.model_wrapper, image, prompt, input_len, inputs, alpha, answer
                )
            finally:
                self.clear_hooks()

