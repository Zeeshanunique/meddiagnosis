from __future__ import annotations

import re

REPORT_PROMPT = (
    "You are reading a chest X-ray. Write a short radiology report in plain English "
    "(4–6 sentences). Include: lung findings, heart/mediastinum, and your clinical impression. "
    "Do not copy instruction numbers or labels like (1), (2), or (3)."
)

FALLBACK_REPORT_PROMPT = (
    "Describe what you see on this chest X-ray. Use full sentences about the lungs, "
    "heart size, and whether there is pneumonia, consolidation, or pleural fluid."
)


def _repetition_ratio(text: str) -> float:
    parts = [p.strip() for p in re.split(r"[.!?]\s+", text) if p.strip()]
    if len(parts) < 2:
        return 0.0
    unique = len(set(parts))
    return 1.0 - (unique / len(parts))


def is_degenerate_findings(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    lower = cleaned.lower()
    if len(cleaned) < 30:
        return True
    if re.search(r"\b1\.\s*2\.\s*3\.", cleaned):
        return True
    if _repetition_ratio(cleaned) > 0.45:
        return True
    junk = (
        "(3) likely diagnosis",
        "(2) likely diagnosis",
        "(1) key findings",
        "line 1 must be",
        "line 2:",
        "pulmonary artery catheter",
    )
    if any(j in lower for j in junk):
        return True
    if lower.startswith("(1)") or lower.startswith("(2)") or lower.startswith("(3)"):
        if len(cleaned) < 80:
            return True
    return False


def fallback_report_from_screen(category_hint: str, screen_label: str) -> str:
    label = screen_label.lower()
    if label == "pneumonia":
        impression = "Findings are suggestive of airspace disease, consistent with pneumonia."
    else:
        impression = "No definite focal consolidation; lungs appear within normal limits for this screen."
    return (
        f"Chest X-ray screen ({category_hint}): {screen_label}. "
        f"{impression} "
        "This is an automated research summary; confirm with a qualified clinician."
    )
