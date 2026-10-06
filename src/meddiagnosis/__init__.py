"""Explainable AI framework for automated medical diagnosis using VLMs."""

import os

# Must be set before torch initialises its MPS backend: lets unsupported ops fall
# back to CPU instead of hard-erroring mid-backward-pass.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
