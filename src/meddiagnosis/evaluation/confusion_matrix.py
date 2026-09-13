from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from meddiagnosis.data.test_dataset import TestDataset, TestSample
from meddiagnosis.evaluation.labels import (
    allowed_labels,
    eval_task_for_sample,
    ground_truth_label,
    parse_label_from_text,
)


@dataclass
class ClassificationRow:
    sample_id: str
    task: str
    true_label: str
    predicted_label: str
    correct: bool
    raw_prediction: str


@dataclass
class ConfusionMatrixResult:
    task: str
    labels: list[str]
    matrix: list[list[int]]
    accuracy: float
    per_class_recall: dict[str, float]
    rows: list[ClassificationRow]


def classify_from_record(sample: TestSample, record: dict) -> ClassificationRow:
    task = eval_task_for_sample(sample)
    true_label = ground_truth_label(sample)
    raw = record.get("findings", "")
    pred = str(record.get("predicted_label", "")).lower() if record.get("predicted_label") else ""
    if not pred:
        pred = parse_label_from_text(raw, task)
    labels = allowed_labels(task)
    if pred not in labels:
        pred = parse_label_from_text(raw, task)
    return ClassificationRow(
        sample_id=sample.id,
        task=task,
        true_label=true_label,
        predicted_label=pred,
        correct=true_label == pred,
        raw_prediction=raw,
    )


def compute_confusion_matrices(
    manifest_path: str | Path,
    results_dir: str | Path,
) -> list[ConfusionMatrixResult]:
    dataset = TestDataset.from_manifest(manifest_path)
    results_dir = Path(results_dir)
    by_task: dict[str, list[ClassificationRow]] = {}

    for sample in dataset.samples:
        path = results_dir / f"{sample.id}_result.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        row = classify_from_record(sample, record)
        by_task.setdefault(row.task, []).append(row)

    results: list[ConfusionMatrixResult] = []
    for task, rows in sorted(by_task.items()):
        labels = list(allowed_labels(task))
        y_true = [r.true_label for r in rows]
        y_pred = [r.predicted_label for r in rows]
        present = sorted(set(y_true) | set(y_pred))
        labels = [l for l in labels if l in present] + [l for l in present if l not in labels]
        idx = {label: i for i, label in enumerate(labels)}
        matrix = [[0] * len(labels) for _ in labels]
        for t, p in zip(y_true, y_pred):
            if t in idx and p in idx:
                matrix[idx[t]][idx[p]] += 1
        total = sum(sum(r) for r in matrix)
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        recall = {
            label: (matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0.0)
            for i, label in enumerate(labels)
        }
        results.append(
            ConfusionMatrixResult(
                task=task,
                labels=labels,
                matrix=matrix,
                accuracy=(correct / total) if total else 0.0,
                per_class_recall=recall,
                rows=rows,
            )
        )
    return results


def save_confusion_matrix_json(results: list[ConfusionMatrixResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_confusion_matrix(result: ConfusionMatrixResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.array(result.matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(max(5, len(result.labels) * 1.2), max(4, len(result.labels))))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(result.labels)))
    ax.set_yticks(range(len(result.labels)))
    ax.set_xticklabels(result.labels, rotation=45, ha="right")
    ax.set_yticklabels(result.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{result.task} (acc={result.accuracy:.2%})")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
