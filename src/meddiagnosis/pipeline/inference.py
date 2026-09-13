from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from meddiagnosis.config import Config
from meddiagnosis.data.test_dataset import TestSample
from meddiagnosis.evaluation.faithfulness import evaluate_grounding
from meddiagnosis.evaluation.labels import (
    _parse_vqa_label,
    eval_task_for_sample,
    ground_truth_label,
    parse_label_from_text,
    reconcile_cxr_labels,
)
from meddiagnosis.evaluation.prompts import (
    build_sample_prompt,
    classifier_system_prompt,
    radiology_cxr_confirm_normal_prompt,
    radiology_cxr_opacity_prompt,
)
from meddiagnosis.evaluation.vqa_helpers import predict_vqa_from_image
from meddiagnosis.models.cxr_classifier import predict_cxr_category
from meddiagnosis.models.smolvlm import SmolVLMLocalModel
from meddiagnosis.xai.gradcam import GradCAMExplainer
from meddiagnosis.xai.visualization import save_attribution


@dataclass
class DiagnosticOutput:
    findings: str
    prompt: str
    gradcam_path: str | None = None
    faithfulness: dict | None = None
    eval_task: str | None = None
    category: str | None = None
    true_label: str | None = None
    predicted_label: str | None = None
    classifier_scores: dict | None = None


class DiagnosticPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        self.model = SmolVLMLocalModel(
            model_id=config.model_id,
            processor_id=config.get("model", "processor_id"),
            system_prompt=config.get("model", "system_prompt"),
        )
        self.gradcam = GradCAMExplainer(
            self.model,
            prefer_fast=bool(config.get("xai", "fast", default=True)),
        )

    @property
    def prompts_cfg(self) -> dict:
        return self.config.get("prompts", default={}) or {}

    def run(
        self,
        image: Image.Image,
        prompt: str,
        sample_id: str = "sample",
        run_xai: bool = True,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> DiagnosticOutput:
        tokens = max_new_tokens or int(
            self.config.get("model", "max_new_tokens", default=128)
        )
        result = self.model.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=tokens,
            system_prompt=system_prompt,
        )
        output = DiagnosticOutput(findings=result.text, prompt=prompt)
        if not run_xai:
            return output

        xai_dir = Path(self.config.get("xai", "output_dir", default="outputs/xai"))
        alpha = float(self.config.get("xai", "alpha", default=0.45))

        print("Running Grad-CAM...", file=sys.stderr)
        t0 = time.perf_counter()
        gradcam = self.gradcam.explain(image, prompt, alpha=alpha)
        print(f"Grad-CAM done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        gradcam_path = xai_dir / f"{sample_id}_gradcam.png"
        save_attribution(str(gradcam_path), gradcam.overlay)
        output.gradcam_path = str(gradcam_path)
        output.faithfulness = asdict(evaluate_grounding(result.text, gradcam.heatmap))
        return output

    def run_classification(
        self,
        sample: TestSample,
        image: Image.Image,
        run_xai: bool = False,
        max_new_tokens: int | None = None,
    ) -> DiagnosticOutput:
        prompt = build_sample_prompt(sample, self.prompts_cfg)
        tokens = max_new_tokens or int(
            self.config.get("model", "classification_max_tokens", default=48)
        )
        sys_prompt = classifier_system_prompt(self.prompts_cfg, self.model.system_prompt)

        output = self.run(
            image=image,
            prompt=prompt,
            sample_id=sample.id,
            run_xai=run_xai,
            max_new_tokens=tokens,
            system_prompt=sys_prompt,
        )

        task = eval_task_for_sample(sample)
        output.eval_task = task
        output.category = sample.category
        output.true_label = ground_truth_label(sample)
        output.predicted_label = parse_label_from_text(output.findings, task)

        if task == "radiology_vqa":
            question = sample.vqa_questions[0] if sample.vqa_questions else ""
            rule_label = predict_vqa_from_image(question, image)
            votes: list[str] = []
            for _ in range(3):
                ans = self.model.generate(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=8,
                    system_prompt=sys_prompt,
                ).text
                label = _parse_vqa_label(ans)
                if label:
                    votes.append(label)
            if rule_label:
                output.predicted_label = rule_label
            elif votes:
                yes_votes = sum(1 for v in votes if v == "yes")
                output.predicted_label = "yes" if yes_votes >= 2 else "no"
            output.findings = (
                f"[vqa_rule: {rule_label or 'none'}] "
                f"[vqa_votes: {','.join(votes) or 'none'}] {output.findings}"
            )

        if task == "radiology_cxr":
            cxr_label, cxr_scores = predict_cxr_category(image)
            output.classifier_scores = cxr_scores
            pneumonia_answer = output.findings
            confirm = self.model.generate(
                image=image,
                prompt=radiology_cxr_confirm_normal_prompt(),
                max_new_tokens=16,
                system_prompt=sys_prompt,
            ).text
            opacity = self.model.generate(
                image=image,
                prompt=radiology_cxr_opacity_prompt(),
                max_new_tokens=16,
                system_prompt=sys_prompt,
            ).text
            vlm_label = reconcile_cxr_labels(pneumonia_answer, confirm, opacity)
            output.predicted_label = cxr_label
            output.findings = (
                f"[cxr_classifier: {cxr_label} score={cxr_scores.get('pneumonia_score', 0):.3f}]\n"
                f"[pneumonia_q: {pneumonia_answer.strip()}]\n"
                f"[normal_q: {confirm.strip()}]\n"
                f"[opacity_q: {opacity.strip()}]\n"
                f"[vlm_vote: {vlm_label}]"
            )

        return output

    def save_result(self, output: DiagnosticOutput, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(output), f, indent=2)
