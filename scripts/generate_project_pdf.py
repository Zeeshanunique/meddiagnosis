#!/usr/bin/env python3
"""Generate MedDiagnosis project documentation PDF."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MedDiagnosis_Project_Documentation.pdf"


class DocPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "MedDiagnosis - Explainable Medical Diagnosis Framework", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(15, 80, 90)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def mono_block(self, text: str) -> None:
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 247, 250)
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(3)


def build_pdf() -> None:
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(10, 70, 80)
    pdf.cell(0, 12, "MedDiagnosis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Explainable AI Framework for Automated Chest X-Ray Diagnosis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5, "Version 0.1.0  |  Research & educational use only - not for clinical diagnosis.")
    pdf.ln(6)

    pdf.chapter_title("1. Project Overview")
    pdf.body(
        "MedDiagnosis is an M.Tech research prototype that combines a lightweight biomedical "
        "Vision-Language Model (VLM) with an explainability layer for chest X-ray analysis. "
        "The system accepts a radiograph and a clinical prompt, generates findings and diagnosis "
        "text, optionally produces visual attribution maps (Grad-CAM / input-gradient saliency), "
        "and evaluates whether the model's explanation aligns with where it visually focused "
        "(faithfulness grounding). A Gradio web interface supports interactive analysis and "
        "follow-up chat on generated reports."
    )

    pdf.chapter_title("2. System Architecture")
    pdf.mono_block(
        "                    CHEST X-RAY IMAGE\n"
        "                           |\n"
        "                           v\n"
        "              +------------------------+\n"
        "              |   Gradio UI / CLI      |\n"
        "              +------------------------+\n"
        "                           |\n"
        "                           v\n"
        "              +------------------------+\n"
        "              |  DiagnosticPipeline    |\n"
        "              +------------------------+\n"
        "                     /            \\\n"
        "                    v              v\n"
        "         +----------------+  +------------------+\n"
        "         | SmolVLMLocal   |  | GradCAMExplainer |\n"
        "         | Model (256M)   |  | + Faithfulness   |\n"
        "         +----------------+  +------------------+\n"
        "                    |              |\n"
        "                    v              v\n"
        "              Findings Text    Heatmap Overlay\n"
        "                    \\              /\n"
        "                     v            v\n"
        "              +------------------------+\n"
        "              |  JSON + PNG outputs    |\n"
        "              +------------------------+"
    )

    pdf.section_title("2.1 Data Flow")
    pdf.body(
        "1. User uploads chest X-ray via CLI or Gradio.\n"
        "2. Image and prompt are tokenized through SmolVLM processor (chat template).\n"
        "3. BMC-smolvlm1-256M generates radiology findings (max 128 tokens default).\n"
        "4. If XAI enabled: input-gradient saliency (Mac) or Grad-CAM (GPU) localizes attention.\n"
        "5. Faithfulness module compares anatomy terms in text vs. heatmap focus region.\n"
        "6. Results saved to outputs/ as JSON and PNG attribution overlays."
    )

    pdf.chapter_title("3. Technology Stack")
    pdf.body(
        "Python 3.10+  |  PyTorch  |  Hugging Face Transformers  |  OpenCV  |  Pillow\n"
        "Model: BIOMEDICA/BMC-smolvlm1-256M (256M params, biomedical VLM)\n"
        "Processor: HuggingFaceTB/SmolVLM-256M-Instruct\n"
        "UI: Gradio 5+ (optional [ui] extra)\n"
        "Runtime: Apple MPS / CUDA / CPU"
    )

    pdf.chapter_title("4. Project Structure")
    pdf.mono_block(
        "meddiagnosis/\n"
        "  config/default.yaml          # Model, prompts, XAI settings\n"
        "  src/meddiagnosis/\n"
        "    cli.py                     # CLI entry (infer, serve)\n"
        "    config.py                  # YAML config loader\n"
        "    models/\n"
        "      smolvlm.py               # Local VLM wrapper\n"
        "      vlm_common.py            # Shared message/token helpers\n"
        "    pipeline/\n"
        "      inference.py             # DiagnosticPipeline orchestrator\n"
        "    xai/\n"
        "      gradcam.py               # Grad-CAM + fast saliency\n"
        "      visualization.py         # Heatmap overlay rendering\n"
        "    evaluation/\n"
        "      faithfulness.py          # Grounding score\n"
        "    app/\n"
        "      gradio_app.py            # Web UI + follow-up chat\n"
        "  outputs/                     # Results and XAI images\n"
        "  pyproject.toml"
    )

    pdf.add_page()
    pdf.chapter_title("5. Module Reference")

    modules = [
        (
            "config.py",
            "Loads config/default.yaml. Exposes model_id, output_dir, ensure_dirs(), "
            "and Hugging Face token setup via HF_TOKEN environment variable.",
        ),
        (
            "models/vlm_common.py",
            "Shared utilities: build_messages(), build_followup_messages(), prepare_inputs(), "
            "generate_from_messages(). Handles chat-template tokenization and device placement.",
        ),
        (
            "models/smolvlm.py - SmolVLMLocalModel",
            "Wraps BIOMEDICA/BMC-smolvlm1-256M. Methods: generate(), chat() for follow-up Q&A, "
            "prepare_inputs() for XAI, get_vision_module() for attribution hooks. "
            "Auto-selects MPS/CUDA/CPU.",
        ),
        (
            "pipeline/inference.py - DiagnosticPipeline",
            "Main orchestrator. Loads model + GradCAMExplainer. run() executes inference and "
            "optional XAI. Returns DiagnosticOutput (findings, gradcam_path, faithfulness). "
            "save_result() writes JSON.",
        ),
        (
            "xai/gradcam.py - GradCAMExplainer",
            "Visual attribution over vision encoder. On Mac (MPS/CPU) uses fast input-gradient "
            "saliency (config xai.fast: true). On CUDA uses full Grad-CAM with hook-based "
            "activation gradients. Fallback to input-gradient on failure.",
        ),
        (
            "xai/visualization.py",
            "normalize_heatmap(), overlay_heatmap() - blends JET colormap heatmap onto X-ray. "
            "save_attribution() writes PNG overlays.",
        ),
        (
            "evaluation/faithfulness.py",
            "Extracts anatomy terms from generated text (lung regions, heart, pleura). Maps "
            "heatmap center-of-mass to anatomical bounding boxes. Computes binary faithfulness "
            "score: 1.0 if predicted focus region appears in explanation, else 0.0.",
        ),
        (
            "app/gradio_app.py",
            "Gradio Blocks UI: image upload, prompt presets, Analyze button, Grad-CAM toggle, "
            "chatbot for follow-up questions on findings, faithfulness display. Preloads model "
            "on page load.",
        ),
        (
            "cli.py",
            "Commands: infer (single-image diagnosis + XAI), serve (launch Gradio on port 7860).",
        ),
    ]
    for name, desc in modules:
        pdf.section_title(name)
        pdf.body(desc)

    pdf.chapter_title("6. Configuration (config/default.yaml)")
    pdf.mono_block(
        "model:\n"
        "  local_id: BIOMEDICA/BMC-smolvlm1-256M\n"
        "  processor_id: HuggingFaceTB/SmolVLM-256M-Instruct\n"
        "  max_new_tokens: 128\n"
        "  system_prompt: Expert radiologist assistant...\n"
        "prompts:\n"
        "  report_generation: Analyze this chest X-ray...\n"
        "xai:\n"
        "  output_dir: outputs/xai\n"
        "  alpha: 0.45          # overlay transparency\n"
        "  fast: true           # fast saliency on Mac\n"
        "paths:\n"
        "  outputs: outputs"
    )

    pdf.chapter_title("7. Usage")
    pdf.section_title("Install")
    pdf.mono_block(
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -e \".[ui]\""
    )
    pdf.section_title("CLI - Fast inference (no XAI)")
    pdf.mono_block(
        "python -m meddiagnosis.cli infer --image chest_xray.png --no-xai --max-tokens 80"
    )
    pdf.section_title("CLI - With explainability (~2-3 min on Mac)")
    pdf.mono_block(
        "python -m meddiagnosis.cli infer --image chest_xray.png --max-tokens 80"
    )
    pdf.section_title("Gradio Web UI")
    pdf.mono_block(
        "python -m meddiagnosis.cli serve --port 7860\n"
        "# Open http://127.0.0.1:7860"
    )

    pdf.chapter_title("8. Outputs")
    pdf.body(
        "outputs/<sample-id>_result.json - findings, prompt, gradcam_path, faithfulness dict\n"
        "outputs/xai/<sample-id>_gradcam.png - heatmap overlay on input X-ray"
    )

    pdf.chapter_title("9. Research Contribution")
    pdf.body(
        "Rather than training a foundation model from scratch, MedDiagnosis leverages a "
        "pre-trained biomedical VLM (BMC-SmolVLM1-256M) and augments it with a multi-level "
        "explainability framework:\n\n"
        "- Visual attribution (Grad-CAM / input-gradient saliency)\n"
        "- Language-level explanation (natural-language findings)\n"
        "- Evidence grounding evaluation (anatomy term vs. heatmap region alignment)\n\n"
        "This enables quantitative assessment of whether generated diagnostic explanations "
        "are faithful to the visual evidence used by the model."
    )

    pdf.chapter_title("10. Limitations")
    pdf.body(
        "- Research prototype only - not validated for clinical use.\n"
        "- BMC-smolvlm trained on general biomedical literature, not chest X-rays specifically.\n"
        "- Faithfulness uses heuristic anatomy bounding boxes, not Chest ImaGenome ground truth.\n"
        "- XAI backward pass is slow on Apple MPS (~2-3 minutes).\n"
        "- Model accuracy depends on prompt quality and image quality."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    build_pdf()
