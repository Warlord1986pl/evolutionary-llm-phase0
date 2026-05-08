from __future__ import annotations
"""Seed stability diagnostic for the Phase 0 base model.

This script tests whether the base model produces effectively identical
outputs across different torch RNG seeds under deterministic generation
settings. The resulting stability check informs whether base-model output is
appropriate as ``seed_text`` for I(X;seed) computation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import unsloth  # Must be imported before transformers for Unsloth patching
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from src.metrics.core import (
    disorganization_entropy,
    effective_complexity,
    shannon_entropy,
)


MODEL_NAME = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"
DIAGNOSTIC_PROMPT = (
    "Summarize the key mechanisms by which misinformation spreads "
    "in online environments and describe evidence-based interventions."
)
TEST_SEEDS = (42, 123, 456, 789, 999)
OUTPUT_PATH = Path("/tmp/seed_stability_results.json")


@dataclass(frozen=True)
class SeedRunResult:
    """Metrics and generated text for one seed-conditioned generation."""

    seed: int
    h_x: float
    c_x: float
    h_dezorg: float
    gen_text: str


def load_model() -> tuple[Any, Any]:
    """Load the Phase 0 base model and tokenizer in 4-bit mode."""
    logging.info("Loading tokenizer for %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logging.info("Loading model in 4-bit mode")
    model, _ = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def generate_text(model: Any, tokenizer: Any, prompt: str, seed: int) -> str:
    """Generate one deterministic response after setting the torch seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    target_device = "cuda" if torch.cuda.is_available() else model.device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def compute_run_result(seed: int, gen_text: str) -> SeedRunResult:
    """Compute metric values for one generated output."""
    return SeedRunResult(
        seed=seed,
        h_x=shannon_entropy(gen_text),
        c_x=effective_complexity(gen_text),
        h_dezorg=disorganization_entropy(gen_text),
        gen_text=gen_text,
    )


def summarize_runs(results: list[SeedRunResult]) -> dict[str, float | str]:
    """Return mean, std, and a stability verdict over all runs."""
    h_x_values = np.array([result.h_x for result in results], dtype=float)
    c_x_values = np.array([result.c_x for result in results], dtype=float)
    h_dezorg_values = np.array([result.h_dezorg for result in results], dtype=float)

    summary = {
        "mean_h_x": float(np.mean(h_x_values)),
        "std_h_x": float(np.std(h_x_values)),
        "mean_c_x": float(np.mean(c_x_values)),
        "std_c_x": float(np.std(c_x_values)),
        "mean_h_dezorg": float(np.mean(h_dezorg_values)),
        "std_h_dezorg": float(np.std(h_dezorg_values)),
    }
    summary["stability_verdict"] = (
        "STABLE"
        if summary["std_h_x"] < 0.1 and summary["std_c_x"] < 0.05
        else "UNSTABLE"
    )
    return summary


def save_results(results: list[SeedRunResult], summary: dict[str, float | str]) -> None:
    """Persist raw outputs and summary statistics to JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": MODEL_NAME,
        "diagnostic_prompt": DIAGNOSTIC_PROMPT,
        "generation": {
            "max_new_tokens": 200,
            "do_sample": False,
            "temperature": 0.0,
        },
        "runs": [asdict(result) for result in results],
        "summary": summary,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved results to %s", OUTPUT_PATH)


def print_results(results: list[SeedRunResult], summary: dict[str, float | str]) -> None:
    """Log per-run metrics and the aggregate stability summary."""
    logging.info("Per-run metrics")
    for result in results:
        preview = result.gen_text[:100].replace("\n", " ")
        logging.info(
            "seed=%s h_x=%.4f c_x=%.4f h_dezorg=%.4f output=%r",
            result.seed,
            result.h_x,
            result.c_x,
            result.h_dezorg,
            preview,
        )

    logging.info(
        "Summary mean_h_x=%.4f std_h_x=%.4f mean_c_x=%.4f std_c_x=%.4f "
        "mean_h_dezorg=%.4f std_h_dezorg=%.4f verdict=%s",
        summary["mean_h_x"],
        summary["std_h_x"],
        summary["mean_c_x"],
        summary["std_c_x"],
        summary["mean_h_dezorg"],
        summary["std_h_dezorg"],
        summary["stability_verdict"],
    )


def main() -> None:
    """Run the seed stability diagnostic and persist the outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model, tokenizer = load_model()

    results: list[SeedRunResult] = []
    for seed in TEST_SEEDS:
        logging.info("Running generation for seed=%s", seed)
        gen_text = generate_text(model=model, tokenizer=tokenizer, prompt=DIAGNOSTIC_PROMPT, seed=seed)
        results.append(compute_run_result(seed=seed, gen_text=gen_text))

    summary = summarize_runs(results)
    print_results(results, summary)
    save_results(results, summary)


if __name__ == "__main__":
    main()