from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from meddiagnosis.evaluation.confusion_matrix import ConfusionMatrixResult

# Reported figures cited from published papers (not reproduced locally).
# Sources, in order:
#  [1] Kermany et al. 2018, "Identifying Medical Diagnoses and Treatable
#      Diseases by Image-Based Deep Learning", Cell 172(5), 1122-1131
#  [2] Chouhan et al. 2020, "A Novel Transfer Learning Based Approach for
#      Pneumonia Detection in Chest X-ray Images", Applied Sciences 10(2), 559
#  [3] Elshennawy & Ibrahim 2020, "Deep-Pneumonia Framework Using Deep
#      Learning Models Based on Chest X-Ray Images", Diagnostics 10(9), 649
#  [4] Stephen et al. 2021 (as reported in MDPI Electronics 10(13), 1512),
#      "Pneumonia Detection from Chest X-ray Images Based on Convolutional
#      Neural Network"
#  [5] Hybrid CNN + Swin Transformer for pneumonia detection, ScienceDirect,
#      2025 (Enhanced Pneumonia Detection in Chest X-rays using Hybrid
#      Convolutional and Vision Transformer Networks)
LITERATURE_COMPARISON: tuple[dict, ...] = (
    {
        "study": "Kermany et al., 2018",
        "technique": "Transfer learning (Inception v3 CNN)",
        "accuracy": 0.928,
        "note": None,
    },
    {
        "study": "Chouhan et al., 2020",
        "technique": "Ensemble (GoogLeNet, ResNet-18, DenseNet-121, Inception v3, AlexNet)",
        "accuracy": 0.964,
        "note": None,
    },
    {
        "study": "Elshennawy & Ibrahim, 2020",
        "technique": "LSTM-CNN",
        "accuracy": 0.91,
        "note": None,
    },
    {
        "study": "Stephen et al., 2021 (MDPI Electronics)",
        "technique": "Custom CNN (trained from scratch)",
        "accuracy": 0.96068,
        "note": None,
    },
    {
        "study": "Hybrid CNN + Swin Transformer, 2025",
        "technique": "Hybrid CNN + Swin Transformer",
        "accuracy": 0.9872,
        "note": None,
    },
)


@dataclass
class ComparisonRow:
    study: str
    technique: str
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    source: str
    n_samples: int | None = None


@dataclass
class ComparisonTable:
    task: str
    rows: list[ComparisonRow]


def build_comparison_table(
    cm_results: list[ConfusionMatrixResult],
    task: str = "radiology_cxr",
    model_name: str = "Proposed model (this project)",
    technique: str = "SmolVLM (vision-language model) + Grad-CAM",
) -> ComparisonTable | None:
    own = next((cm for cm in cm_results if cm.task == task), None)
    if own is None:
        return None

    rows = [
        ComparisonRow(
            study=entry["study"],
            technique=entry["technique"],
            accuracy=entry["accuracy"],
            precision=None,
            recall=None,
            f1=None,
            source="literature" + (f" ({entry['note']})" if entry["note"] else ""),
        )
        for entry in LITERATURE_COMPARISON
    ]
    rows.append(
        ComparisonRow(
            study=model_name,
            technique=technique,
            accuracy=own.accuracy,
            precision=own.macro_precision,
            recall=own.macro_recall,
            f1=own.macro_f1,
            source="measured zero-shot on this project's own test set",
            n_samples=sum(sum(row) for row in own.matrix),
        )
    )
    return ComparisonTable(task=task, rows=rows)


def save_comparison_json(table: ComparisonTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(table), indent=2) + "\n", encoding="utf-8")


def plot_comparison_accuracy(table: ComparisonTable, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row.study for row in table.rows]
    values = [row.accuracy * 100 for row in table.rows]
    colors = ["#4C72B0"] * (len(labels) - 1) + ["#C44E52"]

    own = next(row for row in table.rows if row.precision is not None)
    n = own.n_samples

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5.8))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"NOT A LIKE-FOR-LIKE BENCHMARK ({table.task})", color="#B22222")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", va="bottom")
    fig.text(
        0.5,
        0.015,
        f"Blue: accuracy reported in each paper, from trained classifiers on datasets of thousands of images.\n"
        f"Red: this project, zero-shot (no training) on {n} image(s). Bars are not directly comparable.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison_metrics(table: ComparisonTable, path: Path) -> None:
    import matplotlib.pyplot as plt

    own = next(row for row in table.rows if row.precision is not None)
    metrics = {
        "Accuracy": own.accuracy,
        "Precision": own.precision,
        "Recall": own.recall,
        "F1": own.f1,
    }

    fig, ax = plt.subplots(figsize=(6, 5.4))
    bars = ax.bar(metrics.keys(), [v * 100 for v in metrics.values()], color="#4C72B0")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(f"{own.study} — macro metrics ({table.task})")
    fig.text(
        0.5,
        0.015,
        f"Zero-shot, n={own.n_samples} image(s). Sample far too small for a reliable estimate.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + 1,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
