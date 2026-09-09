#!/usr/bin/env python3
"""Generate academic project documentation PDF for guide presentation."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MedDiagnosis_Project_Documentation.pdf"

TITLE = (
    "Explainable Deep Learning for Multi-Modal Medical Imaging:\n"
    "Toward Trustworthy AI-Assisted Diagnosis in Radiology and Digital Pathology"
)
SUBTITLE = "MedDiagnosis - Research Prototype & Implementation Report"
VERSION = "Version 0.1.0  |  M.Tech Research Project  |  September 2026"


class DocPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, "Explainable Deep Learning for Multi-Modal Medical Imaging", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def chapter(self, num: str, title: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(12, 64, 108)
        self.multi_cell(0, 7, f"{num}  {title}")
        self.ln(1)
        self.set_draw_color(12, 64, 108)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section(self, title: str) -> None:
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def _ensure_space(self, height: float = 20) -> None:
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()

    def body(self, text: str) -> None:
        self._ensure_space(15)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(35, 35, 35)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.2, text)
        self.ln(2)

    def bullets(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(35, 35, 35)
        for item in items:
            self._ensure_space(10)
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.2, f"- {item}")
        self.ln(2)

    def mono(self, text: str) -> None:
        self._ensure_space(20)
        self.set_font("Courier", "", 8.2)
        self.set_fill_color(248, 249, 251)
        self.set_text_color(25, 25, 25)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.3, text, fill=True)
        self.ln(3)

    def info_box(self, label: str, value: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(12, 64, 108)
        self.cell(38, 6, label)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, value)
        self.ln(1)


def _cover(pdf: DocPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(12, 64, 108)
    pdf.rect(0, 0, 210, 52, style="F")
    pdf.ln(18)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 8, "Explainable Deep Learning for\nMulti-Modal Medical Imaging", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Toward Trustworthy AI-Assisted Diagnosis\nin Radiology and Digital Pathology",
        align="C",
    )
    pdf.ln(28)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "MedDiagnosis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, SUBTITLE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, VERSION, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0,
        5,
        "This document describes the motivation, methodology, architecture, and implementation "
        "of an explainable AI framework for medical image analysis. It is intended for academic "
        "review and project guide presentation. Research and educational use only - not for "
        "clinical diagnosis.",
        align="C",
    )


def build_pdf() -> None:
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    _cover(pdf)

    pdf.add_page()
    pdf.chapter("", "Abstract")
    pdf.body(
        "Deep learning has achieved strong performance in medical image analysis, yet black-box "
        "predictions limit clinical adoption because clinicians require transparent, verifiable "
        "reasoning. This project addresses the need for trustworthy AI-assisted diagnosis by "
        "developing MedDiagnosis, an explainable deep learning framework for multi-modal medical "
        "imaging. The system integrates a biomedical Vision-Language Model (VLM) with visual "
        "attribution (Grad-CAM / input-gradient saliency), natural-language report generation, "
        "and a faithfulness evaluation module that checks whether textual explanations align with "
        "model focus regions. The current prototype targets radiology (chest X-ray analysis) as "
        "the primary modality, with a modular architecture designed to extend toward digital "
        "pathology and broader multi-modal fusion. A Gradio interface and CLI enable interactive "
        "demonstration, batch evaluation on a curated test dataset, and follow-up clinical-style "
        "question answering."
    )

    pdf.chapter("1", "Introduction and Background")
    pdf.body(
        "Medical imaging underpins modern diagnosis in radiology (X-ray, CT, MRI) and digital "
        "pathology (whole-slide histopathology). Convolutional neural networks and, more recently, "
        "vision-language models can detect abnormalities, generate reports, and answer visual "
        "questions. However, regulatory bodies and clinical workflows demand interpretability: "
        "clinicians must understand why a model reached a conclusion before trusting it."
    )
    pdf.body(
        "Explainable AI (XAI) methods such as Grad-CAM, saliency maps, and attention visualization "
        "localize model decisions in image space. In radiology, this supports lesion localization "
        "and report verification. In digital pathology, similar techniques can highlight "
        "malignant regions on H&E slides. Multi-modal systems that combine imaging with text "
        "prompts or clinical context further improve usability but increase the need for "
        "cross-modal faithfulness - ensuring language explanations reflect visual evidence."
    )
    pdf.body(
        "MedDiagnosis implements this vision as a research prototype: a lightweight, locally "
        "runnable pipeline that demonstrates end-to-end AI-assisted chest X-ray analysis with "
        "built-in explainability and quantitative grounding checks."
    )

    pdf.chapter("2", "Problem Statement")
    pdf.body(
        "Despite advances in medical deep learning, three barriers prevent trustworthy deployment:"
    )
    pdf.bullets(
        [
            "Opacity: Deep models provide predictions without showing which image regions drove the decision.",
            "Unverified language: VLMs can produce plausible reports not grounded in visual evidence.",
            "Modality silos: Radiology and pathology tools are often separate, limiting unified XAI frameworks.",
        ]
    )
    pdf.body(
        "The core problem this project tackles is: How can we build an explainable, multi-modal "
        "medical imaging system that generates diagnostically useful outputs while providing "
        "visual and textual evidence clinicians can inspect and evaluate?"
    )

    pdf.chapter("3", "Objectives and Scope")
    pdf.section("3.1 Primary Objectives")
    pdf.bullets(
        [
            "Design an explainable deep learning pipeline for medical image diagnosis.",
            "Integrate a biomedical VLM for radiology report generation and visual question answering.",
            "Apply visual attribution (Grad-CAM / saliency) to localize model focus on the input image.",
            "Evaluate explanation faithfulness by comparing anatomy terms in text with heatmap regions.",
            "Provide an interactive demo (Gradio UI) and batch evaluation on a test dataset.",
        ]
    )
    pdf.section("3.2 Project Scope")
    pdf.bullets(
        [
            "Implemented: Chest X-ray (radiology) analysis using BMC-SmolVLM-256M on local Mac/GPU.",
            "Implemented: XAI layer, faithfulness scoring, CLI, Gradio UI, 8-image test dataset.",
            "Planned extension: Digital pathology (H&E whole-slide patches) using the same VLM + XAI pipeline.",
            "Planned extension: Multi-modal fusion (image + clinical notes / lab values) in future phases.",
        ]
    )
    pdf.body(
        "Scope note: The current codebase is a functional radiology prototype. Digital pathology "
        "is addressed architecturally and in future work; it is not yet implemented in code."
    )

    pdf.add_page()
    pdf.chapter("4", "Literature Context")
    pdf.body(
        "This work sits at the intersection of medical AI, explainability, and vision-language modeling:"
    )
    pdf.bullets(
        [
            "Medical VLMs (e.g., MedGemma, PMC-VQA, BIOMEDICA models) combine image understanding with natural-language generation for radiology and biomedical tasks.",
            "Grad-CAM (Selvaraju et al.) and gradient-based saliency provide class-discriminative localization in CNNs and vision encoders.",
            "Faithfulness metrics in XAI assess whether explanations reflect model behavior (same decision under perturbation) and whether text matches saliency.",
            "Benchmarks such as VQA-RAD, MIMIC-CXR, and ChestX-ray14 support evaluation of visual question answering and report generation in radiology.",
            "Digital pathology XAI (patch-level classification + heatmaps on WSIs) parallels radiology attribution but operates at cellular/tissue scale.",
        ]
    )
    pdf.body(
        "MedDiagnosis contributes a practical, lightweight implementation that unifies VLM inference, "
        "visual attribution, and a simple faithfulness heuristic suitable for academic demonstration "
        "on consumer hardware (Apple MPS / CPU)."
    )

    pdf.chapter("5", "Proposed Framework")
    pdf.body(
        "The proposed trustworthy AI-assisted diagnosis framework consists of four layers:"
    )
    pdf.mono(
        "  +----------------------------------------------------------+\n"
        "  |  Layer 4: Trust & Evaluation                             |\n"
        "  |  Faithfulness score, anatomy grounding, batch metrics    |\n"
        "  +----------------------------------------------------------+\n"
        "  |  Layer 3: Explainability (XAI)                         |\n"
        "  |  Grad-CAM, input-gradient saliency, heatmap overlays     |\n"
        "  +----------------------------------------------------------+\n"
        "  |  Layer 2: Multi-Modal Reasoning (VLM)                    |\n"
        "  |  Image + prompt -> findings, diagnosis, VQA responses    |\n"
        "  +----------------------------------------------------------+\n"
        "  |  Layer 1: Input Modalities                               |\n"
        "  |  Radiology (CXR) [implemented] | Pathology [planned]     |\n"
        "  +----------------------------------------------------------+"
    )
    pdf.section("5.1 Radiology Modality (Current Implementation)")
    pdf.body(
        "Chest X-rays are ingested as RGB images. A structured or free-text prompt (e.g., full "
        "report, findings only, VQA question) is passed to the VLM. The model generates "
        "findings and diagnosis text. Optionally, the vision encoder is probed for attribution "
        "maps overlaid on the original radiograph."
    )
    pdf.section("5.2 Digital Pathology Modality (Planned Extension)")
    pdf.body(
        "The same pipeline architecture applies to histopathology: patch or tile images from "
        "H&E-stained slides are fed to the VLM with prompts such as 'Describe tissue morphology' "
        "or 'Is malignancy present?'. Grad-CAM highlights regions of abnormal cellularity. "
        "Modularity in DiagnosticPipeline allows swapping modality-specific prompts and "
        "anatomy/tissue region maps for faithfulness evaluation."
    )
    pdf.section("5.3 Trustworthiness Criteria")
    pdf.bullets(
        [
            "Transparency: User sees both text output and visual heatmap.",
            "Grounding: Faithfulness module checks text-image alignment.",
            "Human-in-the-loop: Gradio chat allows clinician-style follow-up questions.",
            "Reproducibility: Config-driven pipeline, saved JSON/PNG outputs per sample.",
        ]
    )

    pdf.add_page()
    pdf.chapter("6", "System Architecture")
    pdf.mono(
        "   CHEST X-RAY / (future: PATHOLOGY PATCH)\n"
        "                    |\n"
        "                    v\n"
        "         +----------------------+\n"
        "         |   Gradio UI / CLI    |\n"
        "         |  infer | batch | serve|\n"
        "         +----------------------+\n"
        "                    |\n"
        "                    v\n"
        "         +----------------------+\n"
        "         | DiagnosticPipeline   |\n"
        "         +----------------------+\n"
        "              /            \\\n"
        "             v              v\n"
        "   +----------------+  +-------------------+\n"
        "   | SmolVLMLocal   |  | GradCAMExplainer  |\n"
        "   | Model (256M)   |  | + Faithfulness    |\n"
        "   +----------------+  +-------------------+\n"
        "             |                |\n"
        "             v                v\n"
        "      Findings / VQA     Heatmap PNG\n"
        "             \\                /\n"
        "              v              v\n"
        "         +----------------------+\n"
        "         | outputs/ JSON + PNG  |\n"
        "         +----------------------+"
    )
    pdf.section("6.1 Data Flow")
    pdf.bullets(
        [
            "User uploads image via CLI (infer/batch) or Gradio web UI.",
            "Image and prompt are formatted via SmolVLM chat template and tokenized.",
            "BMC-smolvlm1-256M generates radiology findings (default max 128 tokens).",
            "If XAI enabled: input-gradient saliency (Mac MPS/CPU) or Grad-CAM (CUDA) produces heatmap.",
            "Faithfulness module extracts anatomy terms from text and maps heatmap center to lung/heart regions.",
            "Results saved to outputs/ as JSON and attribution PNG; Gradio supports follow-up chat.",
        ]
    )
    pdf.section("6.2 Technology Stack")
    pdf.info_box("Language:", "Python 3.10+")
    pdf.info_box("Deep Learning:", "PyTorch, Hugging Face Transformers")
    pdf.info_box("Model:", "BIOMEDICA/BMC-smolvlm1-256M (256M params, biomedical VLM)")
    pdf.info_box("Processor:", "HuggingFaceTB/SmolVLM-256M-Instruct")
    pdf.info_box("XAI:", "Grad-CAM, input-gradient saliency, OpenCV heatmap overlay")
    pdf.info_box("UI:", "Gradio 5+ (optional [ui] install extra)")
    pdf.info_box("Runtime:", "Apple MPS / CUDA / CPU (~500 MB model, runs locally)")

    pdf.chapter("7", "Implementation Details")
    pdf.section("7.1 Project Structure")
    pdf.mono(
        "meddiagnosis/\n"
        "  config/default.yaml           Model, prompts, XAI settings\n"
        "  data/test_cxr/                Test dataset (8 chest X-rays + manifest)\n"
        "  src/meddiagnosis/\n"
        "    cli.py                      infer, batch, serve commands\n"
        "    config.py                   YAML config loader\n"
        "    data/test_dataset.py        TestDataset loader from manifest\n"
        "    models/smolvlm.py           Local VLM wrapper\n"
        "    models/vlm_common.py        Chat template, tokenization, generation\n"
        "    pipeline/inference.py       DiagnosticPipeline orchestrator\n"
        "    xai/gradcam.py              Grad-CAM + fast saliency (Mac)\n"
        "    xai/visualization.py        Heatmap overlay rendering\n"
        "    evaluation/faithfulness.py  Anatomy grounding score\n"
        "    app/gradio_app.py           Web UI + follow-up chat\n"
        "  scripts/download_test_dataset.py\n"
        "  scripts/generate_project_pdf.py\n"
        "  outputs/                      JSON results + XAI PNGs"
    )
    pdf.section("7.2 Key Modules")
    pdf.bullets(
        [
            "SmolVLMLocalModel: Wraps BMC-smolvlm1-256M; supports generate(), chat(), prepare_inputs() for XAI.",
            "DiagnosticPipeline: Orchestrates inference, optional XAI, faithfulness, and JSON export.",
            "GradCAMExplainer: Hook-based Grad-CAM on CUDA; fast input-gradient saliency on MPS/CPU.",
            "FaithfulnessEvaluator: Maps heatmap focus to anatomy regions; compares with terms in generated text.",
            "TestDataset: Loads manifest.json for batch evaluation with category labels and VQA questions.",
            "Gradio App: Prompt presets, Grad-CAM toggle, example images, interactive follow-up Q&A.",
        ]
    )

    pdf.add_page()
    pdf.chapter("8", "Explainability Methods")
    pdf.section("8.1 Grad-CAM (Gradient-weighted Class Activation Mapping)")
    pdf.body(
        "Grad-CAM computes a coarse localization map by weighting forward-pass activations of the "
        "vision encoder with gradients of the target output (generated token logits) with respect "
        "to those activations. The resulting heatmap highlights regions that most influenced the "
        "model's response. On CUDA GPUs, full Grad-CAM with backward hooks is used."
    )
    pdf.section("8.2 Fast Input-Gradient Saliency (Mac / CPU)")
    pdf.body(
        "Apple MPS backward passes are slow (~2-3 minutes). When config xai.fast is true, the "
        "system uses input-gradient saliency: gradients of the output with respect to input "
        "pixels approximate important regions. This trades some spatial precision for practical "
        "runtime on Mac hardware while preserving explainability for demonstration."
    )
    pdf.section("8.3 Visualization")
    pdf.body(
        "Heatmaps are normalized, color-mapped (JET), and alpha-blended onto the original X-ray "
        "using OpenCV. Overlays are saved to outputs/xai/<sample-id>_gradcam.png for inspection "
        "and inclusion in project reports."
    )

    pdf.chapter("9", "Faithfulness Evaluation")
    pdf.body(
        "Trustworthy AI requires that natural-language explanations reference the same anatomical "
        "regions the model visually attended to. The faithfulness module implements a lightweight "
        "grounding check:"
    )
    pdf.bullets(
        [
            "Extract anatomy terms from generated text (e.g., right upper lung, heart, pleura).",
            "Compute center-of-mass of the saliency heatmap in normalized image coordinates.",
            "Map heatmap focus to predefined anatomical bounding boxes for chest X-rays.",
            "Faithfulness score = 1.0 if predicted focus region appears in mentioned terms, else 0.0.",
        ]
    )
    pdf.body(
        "This heuristic provides a quantitative sanity check for demo and research purposes. "
        "Future work may integrate Chest ImaGenome annotations or human expert ratings for "
        "more rigorous evaluation."
    )

    pdf.chapter("10", "Dataset and Experimental Setup")
    pdf.section("10.1 Test Dataset")
    pdf.body(
        "A curated test set of 8 chest X-ray images is stored in data/test_cxr/images/ with "
        "metadata in manifest.json. Sources include Wikimedia Commons (CC0 / public domain) "
        "and VQA-RAD chest samples (research use)."
    )
    pdf.bullets(
        [
            "3 normal chest X-rays (PA and standard views)",
            "2 pneumonia cases (lobar pneumonia, AP view with upper lobe opacity)",
            "3 VQA-RAD chest images with reference questions and answers",
        ]
    )
    pdf.section("10.2 Download and Batch Evaluation")
    pdf.mono(
        "python scripts/download_test_dataset.py\n"
        "python -m meddiagnosis.cli batch --no-xai --max-tokens 80\n"
        "python -m meddiagnosis.cli batch --limit 3   # subset"
    )
    pdf.section("10.3 Evaluation Metrics")
    pdf.bullets(
        [
            "Qualitative: Visual inspection of findings text and Grad-CAM overlays.",
            "Faithfulness score: Binary grounding alignment per sample.",
            "Latency: Model load time (~10s), inference (~4-5s), XAI (~2-3 min on Mac MPS).",
            "Future: VQA accuracy against VQA-RAD answers, BLEU/ROUGE for report generation.",
        ]
    )

    pdf.add_page()
    pdf.chapter("11", "Demo Workflow and Usage")
    pdf.section("11.1 Installation")
    pdf.mono(
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -e \".[ui]\""
    )
    pdf.section("11.2 Single-Image Inference")
    pdf.mono(
        "# Fast (no XAI)\n"
        "python -m meddiagnosis.cli infer \\\n"
        "  --image data/test_cxr/images/cxr_002_lobar_pneumonia.jpg \\\n"
        "  --no-xai --max-tokens 80\n\n"
        "# With explainability\n"
        "python -m meddiagnosis.cli infer \\\n"
        "  --image data/test_cxr/images/cxr_002_lobar_pneumonia.jpg \\\n"
        "  --max-tokens 80"
    )
    pdf.section("11.3 Interactive Web Demo")
    pdf.mono(
        "python -m meddiagnosis.cli serve --port 7860\n"
        "# Open http://127.0.0.1:7860\n"
        "# Upload X-ray, select prompt preset, toggle Grad-CAM, ask follow-up questions"
    )
    pdf.section("11.4 Outputs")
    pdf.bullets(
        [
            "outputs/<sample-id>_result.json - findings, prompt, gradcam_path, faithfulness dict",
            "outputs/xai/<sample-id>_gradcam.png - heatmap overlay on input image",
        ]
    )

    pdf.chapter("12", "Research Contribution")
    pdf.body(
        "This project demonstrates a practical path toward trustworthy AI-assisted diagnosis by "
        "combining:"
    )
    pdf.bullets(
        [
            "Pre-trained biomedical VLM (no full model training required; suitable for M.Tech timeline).",
            "Multi-level explainability: visual attribution + language explanation + grounding metric.",
            "Modular architecture extensible from radiology to digital pathology and multi-modal inputs.",
            "Local, reproducible deployment on consumer hardware for academic demonstration.",
        ]
    )
    pdf.body(
        "The contribution is methodological and engineering: a unified XAI framework prototype "
        "that a project guide can evaluate through live demo, batch runs, and saved attribution artifacts."
    )

    pdf.chapter("13", "Future Work")
    pdf.bullets(
        [
            "Digital pathology: Apply pipeline to H&E patch datasets (Camelyon, PatchCamelyon).",
            "Multi-modal fusion: Incorporate clinical notes, lab values, or structured EHR fields.",
            "Stronger evaluation: VQA-RAD accuracy, expert radiologist review, Chest ImaGenome grounding.",
            "Model upgrades: MedGemma 4B or domain-fine-tuned CXR models when GPU resources allow.",
            "Integrated Grad-CAM variants: LayerCAM, Score-CAM for comparison studies.",
            "Regulatory awareness: Document bias, uncertainty quantification, and audit trails.",
        ]
    )

    pdf.chapter("14", "Limitations and Ethics")
    pdf.bullets(
        [
            "Research prototype only - not validated for clinical use or regulatory approval.",
            "BMC-smolvlm trained on general biomedical literature, not chest X-rays specifically.",
            "Faithfulness uses heuristic anatomy boxes, not expert-annotated ground truth.",
            "Digital pathology modality is planned but not yet implemented.",
            "XAI backward pass remains slow on Apple MPS; full Grad-CAM preferred on CUDA.",
            "Dataset is small (8 images); conclusions are illustrative, not statistically robust.",
            "AI outputs must not replace qualified medical professionals.",
        ]
    )

    pdf.chapter("15", "Conclusion")
    pdf.body(
        "MedDiagnosis implements an explainable deep learning framework aligned with the theme "
        "'Explainable Deep Learning for Multi-Modal Medical Imaging: Toward Trustworthy "
        "AI-Assisted Diagnosis in Radiology and Digital Pathology.' The radiology prototype "
        "successfully combines VLM-based report generation, visual attribution, and faithfulness "
        "evaluation in an interactive, reproducible system. The modular design provides a clear "
        "foundation for extending to digital pathology and richer multi-modal inputs in subsequent "
        "project phases. This document and the accompanying codebase support demonstration and "
        "review by the project guide."
    )

    pdf.ln(4)
    pdf.section("References (Selected)")
    pdf.bullets(
        [
            "Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks, ICCV 2017.",
            "Lau et al., VQA-RAD: Visual Question Answering in Radiology, 2018.",
            "Johnson et al., MIMIC-CXR: A large publicly available chest X-ray dataset, 2019.",
            "BIOMEDICA, BMC-smolvlm and biomedica_webdataset_24M, Hugging Face, 2024.",
            "Ribeiro et al., Why Should I Trust You? Explaining Predictions of Any Classifier, KDD 2016.",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    build_pdf()
