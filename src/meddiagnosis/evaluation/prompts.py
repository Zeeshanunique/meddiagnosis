from __future__ import annotations

from meddiagnosis.data.test_dataset import TestSample
from meddiagnosis.evaluation.labels import eval_task_for_sample


def build_sample_prompt(sample: TestSample, prompts_cfg: dict) -> str:
    task = eval_task_for_sample(sample)
    if task == "radiology_vqa":
        question = sample.vqa_questions[0] if sample.vqa_questions else "Is there an abnormality?"
        template = prompts_cfg.get(
            "vqa_prompt",
            "Answer with ONLY yes or no.\nQuestion: {question}",
        )
        return template.format(question=question)
    return str(
        prompts_cfg.get(
            "cxr_binary_prompt",
            "Is pneumonia present on this chest X-ray? Answer only yes or no.",
        )
    )


def radiology_cxr_confirm_normal_prompt() -> str:
    return (
        "Are the lungs normal without focal consolidation on this chest X-ray? "
        "Answer with exactly one word: yes or no."
    )


def radiology_cxr_opacity_prompt() -> str:
    return (
        "Is there increased opacity, infiltrate, or consolidation on this chest X-ray? "
        "Answer with exactly one word: yes or no."
    )


def classifier_system_prompt(prompts_cfg: dict, fallback: str | None = None) -> str | None:
    return prompts_cfg.get("classifier_system_prompt") or fallback
