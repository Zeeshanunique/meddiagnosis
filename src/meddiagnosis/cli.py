from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from meddiagnosis.config import Config
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
