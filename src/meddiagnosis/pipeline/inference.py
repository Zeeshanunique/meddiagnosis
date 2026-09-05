from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from meddiagnosis.config import Config
from meddiagnosis.evaluation.faithfulness import evaluate_grounding
from meddiagnosis.models.smolvlm import SmolVLMLocalModel
from meddiagnosis.xai.gradcam import GradCAMExplainer
from meddiagnosis.xai.visualization import save_attribution


@dataclass
class DiagnosticOutput:
    findings: str
    prompt: str
    gradcam_path: str | None = None
    faithfulness: dict | None = None


class DiagnosticPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        self.model = SmolVLMLocalModel(
            model_id=config.model_id,
            processor_id=config.get("model", "processor_id"),
            system_prompt=config.get("model", "system_prompt"),
        )
        self.gradcam = GradCAMExplainer(self.model)

    def run(
        self,
        image: Image.Image,
        prompt: str,
        sample_id: str = "sample",
        run_xai: bool = True,
        max_new_tokens: int | None = None,
    ) -> DiagnosticOutput:
        tokens = max_new_tokens or int(
            self.config.get("model", "max_new_tokens", default=128)
        )
        result = self.model.generate(image=image, prompt=prompt, max_new_tokens=tokens)
        output = DiagnosticOutput(findings=result.text, prompt=prompt)
        if not run_xai:
            return output

        xai_dir = Path(self.config.get("xai", "output_dir", default="outputs/xai"))
        alpha = float(self.config.get("xai", "alpha", default=0.45))

        print("Running Grad-CAM...", file=sys.stderr)
        t0 = time.perf_counter()
        gradcam = self.gradcam.explain(image, prompt, alpha=alpha)
        print(f"Grad-CAM done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        gradcam_path = xai_dir / f"{sample_id}_gradcam.png"
        save_attribution(str(gradcam_path), gradcam.overlay)
        output.gradcam_path = str(gradcam_path)
        output.faithfulness = asdict(evaluate_grounding(result.text, gradcam.heatmap))
        return output

    def save_result(self, output: DiagnosticOutput, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(output), f, indent=2)
