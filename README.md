# MedDiagnosis

Explainable chest X-ray analysis with [BIOMEDICA/BMC-smolvlm1-256M](https://huggingface.co/BIOMEDICA/BMC-smolvlm1-256M): findings, Grad-CAM, faithfulness scoring, and follow-up chat.

**Research use only. Not for clinical diagnosis.**

## Features

- Local inference on Mac MPS / CPU (~500 MB model)
- Grad-CAM + faithfulness grounding check
- Gradio UI with follow-up Q&A on findings

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui]"
```

## Usage

```bash
# Fast (no XAI)
python -m meddiagnosis.cli infer --image chest_xray.png --no-xai --max-tokens 80

# With Grad-CAM (~2–3 min on Mac)
python -m meddiagnosis.cli infer --image chest_xray.png --max-tokens 80

# Web UI + chat
python -m meddiagnosis.cli serve --port 7860
```

## Model training data

SmolVLM was trained on [BIOMEDICA/biomedica_webdataset_24M](https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M) (general biomedical literature, not chest X-rays specifically).

## Config

`config/default.yaml` — model id, prompts, XAI settings.

## Output

`outputs/<sample-id>_result.json` + `outputs/xai/<sample-id>_gradcam.png`
