from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from meddiagnosis.data.test_dataset import TestDataset
from meddiagnosis.evaluation.confusion_matrix import classify_from_record


@dataclass
class EvaluationSummary:
    num_samples: int
    mean_classification_accuracy: float
    samples: list[dict]


def evaluate_results(manifest_path: str | Path, results_dir: str | Path) -> EvaluationSummary:
    dataset = TestDataset.from_manifest(manifest_path)
    results_dir = Path(results_dir)
    rows: list[dict] = []

    for sample in dataset.samples:
        path = results_dir / f"{sample.id}_result.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        cls = classify_from_record(sample, record)
        rows.append(asdict(cls))

    if not rows:
        return EvaluationSummary(num_samples=0, mean_classification_accuracy=0.0, samples=[])

    acc = sum(1 for r in rows if r["correct"]) / len(rows)
    return EvaluationSummary(
        num_samples=len(rows),
        mean_classification_accuracy=acc,
        samples=rows,
    )


def save_evaluation(summary: EvaluationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "num_samples": summary.num_samples,
                "mean_classification_accuracy": summary.mean_classification_accuracy,
                "samples": summary.samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
