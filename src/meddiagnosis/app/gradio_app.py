from __future__ import annotations

import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from meddiagnosis.config import Config
from meddiagnosis.pipeline.inference import DiagnosticPipeline

PROMPT_PRESETS = {
    "Full report": (
        "Analyze this chest X-ray. Provide: (1) key findings, (2) likely diagnosis, "
        "(3) brief explanation of visual evidence."
    ),
    "Findings only": "List the key radiological findings visible in this chest X-ray.",
    "Diagnosis only": "What is the most likely diagnosis based on this chest X-ray?",
    "VQA — lungs normal?": "Are the lungs normal appearing?",
    "VQA — abnormality": "What abnormality is present in this image?",
    "Custom question": "",
}

_pipeline: DiagnosticPipeline | None = None


def _get_pipeline(config_path: str) -> DiagnosticPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DiagnosticPipeline(Config.load(config_path))
    return _pipeline


def _to_pil(image: np.ndarray | Image.Image | None) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return None


def _faithfulness(data: dict | None) -> str:
    if not data:
        return "Enable Grad-CAM for faithfulness score."
    return (
        f"Score {data.get('faithfulness_score', 0):.2f} · "
        f"regions: {', '.join(data.get('mentioned_regions') or []) or 'none'} · "
        f"focus: {data.get('predicted_region') or '—'}"
    )


def _chat_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    return [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant"
    ]


def _findings_message(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}] if text.strip() else []


def analyze(image, preset, custom_prompt, run_xai, max_tokens, config_path):
    empty = [], "", "", _faithfulness(None)
    pil = _to_pil(image)
    if pil is None:
        return empty + ("Waiting for image…",)

    prompt = PROMPT_PRESETS.get(preset, preset)
    if preset == "Custom question":
        prompt = (custom_prompt or "").strip()
        if not prompt:
            return empty + ("Enter a custom question.",)

    t0 = time.perf_counter()
    try:
        out = _get_pipeline(config_path).run(
            image=pil,
            prompt=prompt,
            sample_id=f"ui_{uuid.uuid4().hex[:8]}",
            run_xai=run_xai,
            max_new_tokens=int(max_tokens),
        )
    except Exception as exc:
        return empty + (f"Error ({time.perf_counter() - t0:.1f}s): {exc}",)

    gradcam = Image.open(out.gradcam_path).convert("RGB") if out.gradcam_path else None
    return (
        _findings_message(out.findings),
        gradcam,
        out.findings,
        prompt,
        f"Done in {time.perf_counter() - t0:.1f}s",
        _faithfulness(out.faithfulness),
    )


def chat_followup(image, findings, initial_prompt, chat_history, user_message, max_tokens, config_path):
    message = (user_message or "").strip()
    if not message:
        return chat_history, "", "Type a question below."
    if not (findings or "").strip():
        return chat_history, "", "Click Analyze first."
    pil = _to_pil(image)
    if pil is None:
        return chat_history, "", "Upload a chest X-ray first."

    t0 = time.perf_counter()
    try:
        reply = _get_pipeline(config_path).model.chat(
            pil, initial_prompt, findings, _chat_pairs(chat_history), message, int(max_tokens)
        ).text
    except Exception as exc:
        return chat_history, "", f"Chat failed: {exc}"

    return (
        chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}],
        "",
        f"Replied in {time.perf_counter() - t0:.1f}s",
    )


def build_app(config_path: str = "config/default.yaml") -> gr.Blocks:
    load_dotenv()
    config = Config.load(config_path)

    with gr.Blocks(title="MedDiagnosis") as demo:
        gr.Markdown(f"# MedDiagnosis\n{config.model_id}\n\n*{config.get('disclaimer', 'Research use only.')}*")
        findings_state = gr.State("")
        prompt_state = gr.State("")

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Chest X-ray", type="numpy", height=360)
                preset = gr.Dropdown(list(PROMPT_PRESETS.keys()), value="Full report", label="Preset")
                custom_prompt = gr.Textbox(label="Custom question", visible=False, lines=2)
                run_xai = gr.Checkbox(label="Run Grad-CAM (~2–3 min on Mac)", value=False)
                max_tokens = gr.Slider(32, 256, value=int(config.get("model", "max_new_tokens", default=128)), step=16)
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                chatbot = gr.Chatbot(label="Findings & follow-up", height=420)
                chat_input = gr.Textbox(
                    placeholder="Ask a follow-up question about the report…",
                    show_label=False,
                    lines=2,
                )
                chat_btn = gr.Button("Send", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                gradcam_out = gr.Image(label="Grad-CAM", type="pil", height=220)
                faithfulness = gr.Textbox(label="Faithfulness", interactive=False)

        example_paths = [
            str(p)
            for p in sorted((Path("data/test_cxr/images").glob("*")))
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not example_paths and Path("chest_xray.png").exists():
            example_paths = ["chest_xray.png"]
        if example_paths:
            gr.Examples(example_paths[:8], inputs=[image_in], label="Test dataset samples")

        preset.change(lambda p: gr.update(visible=p == "Custom question"), preset, custom_prompt)
        analyze_btn.click(
            analyze,
            [image_in, preset, custom_prompt, run_xai, max_tokens, gr.State(config_path)],
            [chatbot, gradcam_out, findings_state, prompt_state, status, faithfulness],
        )

        chat_args = [image_in, findings_state, prompt_state, chatbot, chat_input, max_tokens, gr.State(config_path)]
        chat_btn.click(chat_followup, chat_args, [chatbot, chat_input, status])
        chat_input.submit(chat_followup, chat_args, [chatbot, chat_input, status])
        chatbot.clear(
            lambda f: (_findings_message(f), ""),
            inputs=[findings_state],
            outputs=[chatbot, chat_input],
        )

        def _preload():
            _get_pipeline(config_path)
            return f"Ready · {config.model_id}"

        demo.load(_preload, outputs=status)

    return demo


def launch(config_path: str = "config/default.yaml", server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
    build_app(config_path).launch(server_name=server_name, server_port=server_port, share=share)
