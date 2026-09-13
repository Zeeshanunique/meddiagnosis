from __future__ import annotations

import numpy as np
from PIL import Image


def _heuristic_pneumonia_score(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    left = gray[:, int(w * 0.05) : int(w * 0.45)]
    right = gray[:, int(w * 0.55) : int(w * 0.95)]
    return float(abs(left.std() - right.std()) + 0.5 * (left.std() + right.std()))


def predict_cxr_category(image: Image.Image, threshold: float = 0.205) -> tuple[str, dict[str, float]]:
    score = _heuristic_pneumonia_score(image)
    label = "pneumonia" if score >= threshold else "normal"
    return label, {"pneumonia_score": score, "classifier_source": "heuristic"}
