from __future__ import annotations

import re

from meddiagnosis.data.test_dataset import TestSample

CXR_LABELS = ("normal", "pneumonia")
VQA_LABELS = ("yes", "no")

CXR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pneumonia": ("pneumonia", "consolidation", "infiltrate", "opacity", "airspace", "lobar"),
    "normal": ("normal", "unremarkable", "no acute", "clear lungs", "no consolidation"),
}

VQA_YES = ("yes", "present", "evidence", "seen", "consistent with", "suggestive")
VQA_NO = ("no", "not present", "absent", "without", "none", "unremarkable", "no evidence")


def eval_task_for_sample(sample: TestSample) -> str:
    return "radiology_vqa" if sample.category == "vqa_rad" else "radiology_cxr"


def ground_truth_label(sample: TestSample) -> str:
    if sample.category == "vqa_rad":
        text = sample.reference_hint.strip().lower()
        return "yes" if text.startswith("y") else "no"
    return sample.category.strip().lower()


def allowed_labels(task: str) -> tuple[str, ...]:
    return VQA_LABELS if task == "radiology_vqa" else CXR_LABELS


def _parse_vqa_label(text: str) -> str | None:
    cleaned = text.strip().lower()
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    if any(p in cleaned for p in VQA_NO) and not any(p in cleaned for p in VQA_YES):
        return "no"
    if any(p in cleaned for p in VQA_YES):
        return "yes"
    return None


def reconcile_cxr_labels(
    pneumonia_answer: str,
    normal_answer: str,
    opacity_answer: str | None = None,
) -> str:
    votes: list[str] = []
    p_raw = _parse_vqa_label(pneumonia_answer)
    n_raw = _parse_vqa_label(normal_answer)
    o_raw = _parse_vqa_label(opacity_answer or "")

    if p_raw == "yes":
        votes.append("pneumonia")
    elif p_raw == "no":
        votes.append("normal")
    if n_raw == "yes":
        votes.append("normal")
    elif n_raw == "no":
        votes.append("pneumonia")
    if o_raw == "yes":
        votes.append("pneumonia")
    elif o_raw == "no":
        votes.append("normal")

    if not votes:
        return "normal"
    pneumonia_votes = sum(1 for v in votes if v == "pneumonia")
    normal_votes = sum(1 for v in votes if v == "normal")
    if pneumonia_votes > normal_votes:
        return "pneumonia"
    if normal_votes > pneumonia_votes:
        return "normal"
    return votes[-1]


def parse_label_from_text(text: str, task: str) -> str:
    if task == "radiology_vqa":
        return _parse_vqa_label(text) or "no"
    if task == "radiology_cxr":
        text_lower = text.lower()
        scores = {label: 0 for label in CXR_LABELS}
        for label, words in CXR_KEYWORDS.items():
            for word in words:
                if word in text_lower:
                    scores[label] += 1
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "normal"
    return "normal"
