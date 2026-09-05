from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

ANATOMY_REGIONS = {
    "right upper lung": (0.55, 0.05, 0.95, 0.35),
    "left upper lung": (0.05, 0.05, 0.45, 0.35),
    "right lower lung": (0.55, 0.45, 0.95, 0.90),
    "left lower lung": (0.05, 0.45, 0.45, 0.90),
    "right lung": (0.50, 0.05, 0.95, 0.90),
    "left lung": (0.05, 0.05, 0.50, 0.90),
    "heart": (0.35, 0.35, 0.65, 0.70),
    "mediastinum": (0.35, 0.20, 0.65, 0.55),
    "pleura": (0.05, 0.05, 0.95, 0.90),
}


@dataclass
class GroundingResult:
    mentioned_regions: list[str]
    predicted_region: str | None
    faithfulness_score: float


def extract_anatomy_terms(text: str) -> list[str]:
    text = text.lower()
    found = []
    for region in sorted(ANATOMY_REGIONS, key=len, reverse=True):
        if region in text and region not in found:
            found.append(region)
    return found


def region_from_heatmap(heatmap: np.ndarray) -> str | None:
    hm = heatmap.astype(np.float64)
    hm = hm - hm.min()
    total = hm.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(hm.shape)
    cy = (ys * hm).sum() / total / hm.shape[0]
    cx = (xs * hm).sum() / total / hm.shape[1]
    for region, (x1, y1, x2, y2) in ANATOMY_REGIONS.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return region
    return None


def evaluate_grounding(explanation: str, heatmap: np.ndarray) -> GroundingResult:
    mentioned = extract_anatomy_terms(explanation)
    predicted = region_from_heatmap(heatmap)
    score = 1.0 if predicted and predicted in mentioned else 0.0
    return GroundingResult(
        mentioned_regions=mentioned,
        predicted_region=predicted,
        faithfulness_score=score,
    )


if __name__ == "__main__":
    assert "pleura" in extract_anatomy_terms("bilateral pleural effusion")
    assert evaluate_grounding("heart enlarged", np.zeros((8, 8))).faithfulness_score == 0.0
