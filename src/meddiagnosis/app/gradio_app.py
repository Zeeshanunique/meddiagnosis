from __future__ import annotations

import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from meddiagnosis.config import Config
from meddiagnosis.evaluation.comparison import (
    build_comparison_table,
    plot_comparison_accuracy,
    plot_comparison_metrics,
)
from meddiagnosis.evaluation.confusion_matrix import compute_confusion_matrices, plot_confusion_matrix
from meddiagnosis.pipeline.inference import DiagnosticPipeline

PROMPT_PRESETS = {
    "Full report": (
        "Write a short chest X-ray report in plain English (4-6 sentences): lung findings, "
        "heart/mediastinum, and clinical impression. Do not use numbered labels (1)(2)(3)."
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


def run_evaluation(manifest_path: str, results_dir: str):
    cm_results = compute_confusion_matrices(manifest_path, results_dir)
    if not cm_results:
        empty_msg = "No results found. Run `meddiagnosis batch --mode classification` first."
        return empty_msg, None, None, [], None, None

    rows = []
    cm_images = []
    for cm in cm_results:
        for label in cm.labels:
            rows.append(
                [
                    cm.task,
                    label,
                    f"{cm.per_class_precision[label]:.2%}",
                    f"{cm.per_class_recall[label]:.2%}",
                    f"{cm.per_class_f1[label]:.2%}",
                ]
            )
        rows.append(
            [cm.task, "macro avg", f"{cm.macro_precision:.2%}", f"{cm.macro_recall:.2%}", f"{cm.macro_f1:.2%}"]
        )
        png = Path(results_dir) / f"confusion_matrix_{cm.task}.png"
        plot_confusion_matrix(cm, png)
        cm_images.append(str(png))

    summary = "  |  ".join(f"{cm.task}: accuracy={cm.accuracy:.2%}" for cm in cm_results)

    table = build_comparison_table(cm_results)
    comparison_acc_png = comparison_metrics_png = None
    if table:
        comparison_acc_png = Path(results_dir) / "comparison_accuracy.png"
        comparison_metrics_png = Path(results_dir) / "comparison_metrics.png"
        plot_comparison_accuracy(table, comparison_acc_png)
        plot_comparison_metrics(table, comparison_metrics_png)

    return (
        summary,
        cm_images[0] if len(cm_images) > 0 else None,
        cm_images[1] if len(cm_images) > 1 else None,
        rows,
        str(comparison_acc_png) if comparison_acc_png else None,
        str(comparison_metrics_png) if comparison_metrics_png else None,
    )


def build_app(config_path: str = "config/default.yaml") -> gr.Blocks:
    load_dotenv()
    config = Config.load(config_path)

    with gr.Blocks(title="MedDiagnosis") as demo:
        gr.Markdown(f"# MedDiagnosis\n{config.model_id}\n\n*{config.get('disclaimer', 'Research use only.')}*")
        findings_state = gr.State("")
        prompt_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("Diagnose"):
                with gr.Row():
                    with gr.Column():
                        image_in = gr.Image(label="Chest X-ray", type="numpy", height=360)
                        preset = gr.Dropdown(list(PROMPT_PRESETS.keys()), value="Full report", label="Preset")
                        custom_prompt = gr.Textbox(label="Custom question", visible=False, lines=2)
                        run_xai = gr.Checkbox(label="Run Grad-CAM (~45-60s on Mac)", value=False)
                        max_tokens = gr.Slider(
                            32, 256, value=int(config.get("model", "max_new_tokens", default=128)), step=16
                        )
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

                chat_args = [
                    image_in,
                    findings_state,
                    prompt_state,
                    chatbot,
                    chat_input,
                    max_tokens,
                    gr.State(config_path),
                ]
                chat_btn.click(chat_followup, chat_args, [chatbot, chat_input, status])
                chat_input.submit(chat_followup, chat_args, [chatbot, chat_input, status])
                chatbot.clear(
                    lambda f: (_findings_message(f), ""),
                    inputs=[findings_state],
                    outputs=[chatbot, chat_input],
                )

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "Metrics computed from `meddiagnosis batch --mode classification` results in `outputs/`.\n\n"
                    "⚠️ **Zero-shot, n=5 CXR + 3 VQA images.** The model is used as-is with no training or "
                    "fine-tuning on chest X-rays. A sample this small cannot support a reliable estimate, and "
                    "these numbers are **not** comparable to published benchmarks trained on thousands of images."
                )
                with gr.Row():
                    manifest_in = gr.Textbox(value="data/test_cxr/manifest.json", label="Manifest path")
                    results_dir_in = gr.Textbox(value=config.output_dir, label="Results directory")
                    eval_btn = gr.Button("Run evaluation", variant="primary")
                eval_summary = gr.Textbox(label="Accuracy summary", interactive=False)
                metrics_table = gr.Dataframe(
                    headers=["Task", "Label", "Precision", "Recall", "F1"],
                    label="Precision / Recall / F1 by class",
                )
                with gr.Row():
                    cm_cxr_out = gr.Image(label="Confusion matrix — radiology_cxr", type="filepath")
                    cm_vqa_out = gr.Image(label="Confusion matrix — radiology_vqa", type="filepath")
                with gr.Row():
                    comparison_acc_out = gr.Image(label="Accuracy vs. published studies", type="filepath")
                    comparison_metrics_out = gr.Image(label="This model's metrics", type="filepath")

                eval_btn.click(
                    run_evaluation,
                    [manifest_in, results_dir_in],
                    [
                        eval_summary,
                        cm_cxr_out,
                        cm_vqa_out,
                        metrics_table,
                        comparison_acc_out,
                        comparison_metrics_out,
                    ],
                )

        def _preload():
            _get_pipeline(config_path)
            return f"Ready · {config.model_id}"

        demo.load(_preload, outputs=status)

    return demo


def launch(config_path: str = "config/default.yaml", server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
    build_app(config_path).launch(server_name=server_name, server_port=server_port, share=share)
