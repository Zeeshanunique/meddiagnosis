#!/usr/bin/env python3
"""Download public-domain chest X-ray images for local testing."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "test_cxr" / "images"
MANIFEST = ROOT / "data" / "test_cxr" / "manifest.json"

# CC0 / US public domain sources (Wikimedia Commons)
WIKIMEDIA_FILES: list[dict] = [
    {
        "id": "cxr_001_normal_pa",
        "filename": "cxr_001_normal_pa.jpg",
        "title": "File:Normal posteroanterior (PA) chest radiograph (X-ray).jpg",
        "category": "normal",
        "reference_hint": "Normal PA chest radiograph",
        "vqa_questions": [
            "Are the lungs normal appearing?",
            "Is there any focal consolidation?",
        ],
    },
    {
        "id": "cxr_002_lobar_pneumonia",
        "filename": "cxr_002_lobar_pneumonia.jpg",
        "title": "File:X-ray of lobar pneumonia.jpg",
        "category": "pneumonia",
        "reference_hint": "Lobar pneumonia, right middle lobe",
        "vqa_questions": [
            "What abnormality is present in this image?",
            "Is there pneumonia?",
        ],
    },
    {
        "id": "cxr_003_pneumonia_ap",
        "filename": "cxr_003_pneumonia_ap.jpg",
        "title": "File:Pneumonia x ray.jpg",
        "category": "pneumonia",
        "reference_hint": "Pneumonia with right upper lobe opacity (CDC PHIL)",
        "vqa_questions": [
            "Is there increased opacity in the lungs?",
            "Which lobe shows abnormality?",
        ],
    },
    {
        "id": "cxr_004_normal",
        "filename": "cxr_004_normal.png",
        "title": "File:Chest.png",
        "category": "normal",
        "reference_hint": "Normal heart and lungs",
        "vqa_questions": [
            "Are the lungs normal appearing?",
            "Is the heart size normal?",
        ],
    },
    {
        "id": "cxr_005_normal_pa",
        "filename": "cxr_005_normal_pa.png",
        "title": "File:Chest Xray PA 3-8-2010.png",
        "category": "normal",
        "reference_hint": "PA chest radiograph",
        "vqa_questions": [
            "Describe the key findings in this chest X-ray.",
        ],
    },
]


_UA = "MedDiagnosis/0.1 (research; +https://github.com/meddiagnosis)"


def _open(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)


def _wikimedia_download_url(title: str) -> str:
    api = (
        "https://commons.wikimedia.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            }
        )
    )
    with _open(api, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise FileNotFoundError(f"Wikimedia file not found: {title}")
    return page["imageinfo"][0]["url"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _open(url, timeout=120) as resp, dest.open("wb") as f:
        f.write(resp.read())


def _try_vqa_rad(limit: int = 3) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Optional: pip install datasets to add VQA-RAD chest samples")
        return []

    try:
        ds = load_dataset("abhay2812/vqa-rad", split="test")
    except Exception as exc:
        print(f"Optional VQA-RAD download skipped: {exc}")
        return []
    entries: list[dict] = []
    seen: set[str] = set()
    idx = 0
    for row in ds:
        organ = (row.get("image_organ") or row.get("modality") or "").upper()
        if organ != "CHEST":
            continue
        image = row.get("image")
        if image is None:
            continue
        key = str(row.get("image_name") or idx)
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        filename = f"cxr_vqa_{idx:03d}.jpg"
        path = OUT_DIR / filename
        image.convert("RGB").save(path, quality=95)
        question = row.get("question") or ""
        answer = row.get("answer") or ""
        entries.append(
            {
                "id": f"cxr_vqa_{idx:03d}",
                "filename": filename,
                "category": "vqa_rad",
                "source": "abhay2812/vqa-rad",
                "reference_hint": answer,
                "vqa_questions": [question] if question else [],
                "license": "VQA-RAD (research use; cite original paper)",
            }
        )
        if len(entries) >= limit:
            break
    return entries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for spec in WIKIMEDIA_FILES:
        dest = OUT_DIR / spec["filename"]
        print(f"Downloading {spec['id']} …")
        try:
            url = _wikimedia_download_url(spec["title"])
            _download(url, dest)
        except Exception as exc:
            print(f"  SKIP {spec['id']}: {exc}")
            continue
        manifest.append(
            {
                "id": spec["id"],
                "filename": spec["filename"],
                "path": str(dest.relative_to(ROOT)),
                "category": spec["category"],
                "source": "Wikimedia Commons",
                "reference_hint": spec["reference_hint"],
                "vqa_questions": spec["vqa_questions"],
                "license": "CC0 / Public Domain",
            }
        )
        print(f"  -> {dest}")

    vqa_entries = _try_vqa_rad(limit=3)
    manifest.extend(vqa_entries)
    for entry in vqa_entries:
        print(f"  VQA-RAD -> {OUT_DIR / entry['filename']}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"samples": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(manifest)} samples to {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
