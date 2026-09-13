from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from meddiagnosis.config import Config
from meddiagnosis.data.test_dataset import TestDataset
from meddiagnosis.evaluation.confusion_matrix import (
    compute_confusion_matrices,
    plot_confusion_matrix,
    save_confusion_matrix_json,
)
from meddiagnosis.evaluation.metrics import evaluate_results, save_evaluation
from meddiagnosis.pipeline.inference import DiagnosticPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable chest X-ray diagnosis (SmolVLM + Grad-CAM)"
    )
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="Run diagnosis + XAI on one image")
    infer.add_argument("--image", required=True, help="Path to chest X-ray image")
    infer.add_argument("--prompt", default=None, help="Custom prompt")
    infer.add_argument("--sample-id", default="demo")
    infer.add_argument("--no-xai", action="store_true")
    infer.add_argument("--max-tokens", type=int, default=None, help="Override max_new_tokens")

    batch = sub.add_parser("batch", help="Run inference on the test dataset manifest")
    batch.add_argument(
        "--manifest",
        default="data/test_cxr/manifest.json",
        help="Path to test dataset manifest.json",
    )
    batch.add_argument("--prompt", default=None, help="Custom prompt (overrides VQA questions)")
    batch.add_argument("--no-xai", action="store_true")
    batch.add_argument("--max-tokens", type=int, default=None)
    batch.add_argument("--limit", type=int, default=None, help="Max number of samples to run")
    batch.add_argument(
        "--mode",
        choices=["classification", "report"],
        default="classification",
        help="classification: tuned labels; report: free-text findings",
    )

    evaluate = sub.add_parser("evaluate", help="Classification metrics and confusion matrix")
    evaluate.add_argument("--manifest", default="data/test_cxr/manifest.json")
    evaluate.add_argument("--results-dir", default="outputs")
    evaluate.add_argument("--output", default=None)

    benchmark = sub.add_parser("benchmark", help="Run classification batch + evaluate")
    benchmark.add_argument("--manifest", default="data/test_cxr/manifest.json")
    benchmark.add_argument("--no-xai", action="store_true")
    benchmark.add_argument("--limit", type=int, default=None)

    serve = sub.add_parser("serve", help="Launch Gradio web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = Config.load(args.config)

    if args.command == "infer":
        image = Image.open(args.image).convert("RGB")
        prompt = args.prompt or config.get("prompts", "report_generation")
        pipeline = DiagnosticPipeline(config)
        output = pipeline.run(
            image=image,
            prompt=prompt,
            sample_id=args.sample_id,
            run_xai=not args.no_xai,
            max_new_tokens=args.max_tokens,
        )
        out_path = Path(config.output_dir) / f"{args.sample_id}_result.json"
        pipeline.save_result(output, out_path)
        print(output.findings)
        if output.faithfulness:
            print(f"Faithfulness: {output.faithfulness['faithfulness_score']:.3f}")
        print(f"Saved: {out_path}")

    elif args.command == "batch":
        dataset = TestDataset.from_manifest(args.manifest)
        samples = dataset.samples[: args.limit] if args.limit else dataset.samples
        if not samples:
            raise SystemExit(f"No samples in {args.manifest}. Run: python scripts/download_test_dataset.py")

        pipeline = DiagnosticPipeline(config)
        default_prompt = args.prompt or config.get("prompts", "report_generation")
        print(f"Running batch ({args.mode}) on {len(samples)} sample(s)…\n")

        for sample in samples:
            image = dataset.load_image(sample)
            if args.mode == "classification":
                output = pipeline.run_classification(
                    sample=sample,
                    image=image,
                    run_xai=not args.no_xai,
                    max_new_tokens=args.max_tokens,
                )
            else:
                prompt = default_prompt
                if args.prompt is None and sample.vqa_questions:
                    prompt = sample.vqa_questions[0]
                output = pipeline.run(
                    image=image,
                    prompt=prompt,
                    sample_id=sample.id,
                    run_xai=not args.no_xai,
                    max_new_tokens=args.max_tokens,
                )
            out_path = Path(config.output_dir) / f"{sample.id}_result.json"
            pipeline.save_result(output, out_path)
            print(f"[{sample.id}] {sample.category} — {sample.reference_hint}")
            if output.predicted_label and output.true_label:
                mark = "OK" if output.predicted_label == output.true_label else "MISS"
                print(f"  label: {output.true_label} -> {output.predicted_label} [{mark}]")
            print(output.findings[:200] + ("…" if len(output.findings) > 200 else ""))
            print(f"  saved: {out_path}\n")

    elif args.command == "evaluate":
        summary = evaluate_results(args.manifest, args.results_dir)
        out = Path(args.output or config.output_dir) / "evaluation.json"
        save_evaluation(summary, out)
        print(f"Samples: {summary.num_samples}")
        print(f"Classification accuracy: {summary.mean_classification_accuracy:.3f}")
        print(f"Saved: {out}")

        cm_results = compute_confusion_matrices(args.manifest, args.results_dir)
        if cm_results:
            cm_json = Path(config.output_dir) / "confusion_matrix.json"
            save_confusion_matrix_json(cm_results, cm_json)
            for cm in cm_results:
                png = Path(config.output_dir) / f"confusion_matrix_{cm.task}.png"
                plot_confusion_matrix(cm, png)
                print(f"\n{cm.task}: accuracy={cm.accuracy:.2%}")
                for label, recall in cm.per_class_recall.items():
                    print(f"  recall[{label}]: {recall:.2%}")
                print(f"  PNG: {png}")
            print(f"JSON: {cm_json}")
        else:
            print("No results found. Run: meddiagnosis batch --mode classification")

    elif args.command == "benchmark":
        dataset = TestDataset.from_manifest(args.manifest)
        samples = dataset.samples[: args.limit] if args.limit else dataset.samples
        pipeline = DiagnosticPipeline(config)
        print(f"Benchmark: classification on {len(samples)} sample(s)…\n")
        for sample in samples:
            image = dataset.load_image(sample)
            output = pipeline.run_classification(sample=sample, image=image, run_xai=not args.no_xai)
            pipeline.save_result(output, Path(config.output_dir) / f"{sample.id}_result.json")
        args.manifest = args.manifest
        args.results_dir = str(config.output_dir)
        args.output = None
        # Re-use evaluate branch logic inline
        summary = evaluate_results(args.manifest, args.results_dir)
        save_evaluation(summary, Path(config.output_dir) / "evaluation.json")
        cm_results = compute_confusion_matrices(args.manifest, args.results_dir)
        save_confusion_matrix_json(cm_results, Path(config.output_dir) / "confusion_matrix.json")
        for cm in cm_results:
            plot_confusion_matrix(cm, Path(config.output_dir) / f"confusion_matrix_{cm.task}.png")
        print(f"Accuracy: {summary.mean_classification_accuracy:.2%}")

    elif args.command == "serve":
        from meddiagnosis.app.gradio_app import launch

        launch(
            config_path=args.config,
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
