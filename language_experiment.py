from __future__ import annotations

from pathlib import Path
import csv
import json
import random
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from language_models import (
    RELATION_LABELS,
    RELATION_PHRASES,
    train_language_models,
    evaluate_language_by_length,
    evaluate_language_contradictions,
)


def save_csv(path: Path, results: dict) -> None:
    keys = list(results.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(results[k] for k in keys)))


def save_plot(path: Path, results: dict, title: str, ylabel: str, train_max_len: int):
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_subplot(111)
    ax.plot(
        results["length"],
        results["phase"],
        marker="o",
        markersize=3,
        label="Language → relational phase",
    )
    ax.plot(
        results["length"],
        results["transformer"],
        marker="o",
        markersize=3,
        label="Tiny transformer",
    )
    ax.axvline(
        train_max_len,
        linestyle="--",
        label=f"Training length limit = {train_max_len}",
    )
    ax.set_title(title)
    ax.set_xlabel("Number of language relations")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(results: dict, train_max_len: int):
    x = np.asarray(results["length"])
    in_mask = x <= train_max_len
    out_mask = x > train_max_len
    summary = {}
    for key in ("phase", "transformer"):
        y = np.asarray(results[key], dtype=float)
        summary[key] = {
            "mean_in_distribution": float(y[in_mask].mean()) if in_mask.any() else None,
            "mean_out_of_distribution": float(y[out_mask].mean()) if out_mask.any() else None,
            "last_length_accuracy": float(y[-1]),
        }
    return summary


def make_sample_examples(path: Path):
    text = """EXPERIMENT 1 — SAMPLE LANGUAGE CHAINS

Latent relation algebra:
  aligned = 0
  clockwise quarter-turn = +1
  opposite = +2
  counterclockwise quarter-turn = +3
(all modulo 4)

The neural models do not receive the numeric relation for an input phrase.
They receive the words.

Example:
  B is a quarter turn clockwise from A.
  C is opposite to B.
  D is turned left ninety degrees from C.

The target asks for D relative to A.

The language→phase model learns a unit complex relation for each phrase,
then composes those learned relations structurally by complex multiplication.

The transformer receives the same phrase words but no explicit composition rule.
"""
    path.write_text(text, encoding="utf-8")


def run_language_experiment(
    *,
    output_dir: str | Path = "language_results",
    seed: int = 11,
    train_max_len: int = 5,
    test_max_len: int = 32,
    steps: int = 900,
    batch_size: int = 128,
    examples_per_length: int = 400,
    progress_callback: Optional[Callable] = None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase_model, transformer, history = train_language_models(
        seed=seed,
        train_max_len=train_max_len,
        test_max_len=test_max_len,
        steps=steps,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    generalization = evaluate_language_by_length(
        phase_model,
        transformer,
        min_len=1,
        max_len=test_max_len,
        examples_per_length=examples_per_length,
        seed=seed + 1000,
    )

    contradictions = evaluate_language_contradictions(
        phase_model,
        transformer,
        min_len=3,
        max_len=test_max_len,
        examples_per_length=examples_per_length,
        seed=seed + 2000,
    )

    save_csv(out / "language_generalization.csv", generalization)
    save_csv(out / "language_contradictions.csv", contradictions)

    save_plot(
        out / "language_generalization.png",
        generalization,
        "Experiment 1: Language → Relational Geometry → Composition",
        "Exact relation accuracy",
        train_max_len,
    )
    save_plot(
        out / "language_contradictions.png",
        contradictions,
        "Experiment 1: Language Loop Closure / Contradiction Detection",
        "Consistent / inconsistent accuracy",
        train_max_len,
    )

    phrase_table = phase_model.phrase_phase_table()

    report = {
        "experiment": "Language-to-geometry compositional generalization",
        "seed": seed,
        "train_max_len": train_max_len,
        "test_max_len": test_max_len,
        "steps": steps,
        "batch_size": batch_size,
        "examples_per_length": examples_per_length,
        "relation_phrases": RELATION_PHRASES,
        "phrase_phase_table": phrase_table,
        "generalization_summary": summarize(generalization, train_max_len),
        "contradiction_summary": summarize(contradictions, train_max_len),
        "interpretation_warning": (
            "This remains a controlled synthetic language experiment. "
            "The relation vocabulary is small and the relational model is given "
            "complex multiplication as its composition rule. The experiment tests "
            "whether it can learn language-to-relation mappings from end-to-end labels "
            "and preserve systematic composition beyond the training depth. It does "
            "not establish performance on unrestricted natural language."
        ),
    }
    (out / "language_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_sample_examples(out / "sample_examples.txt")

    torch.save(
        {
            "phase_model": phase_model.state_dict(),
            "transformer": transformer.state_dict(),
            "report": report,
        },
        out / "language_models.pt",
    )

    return {
        "report": report,
        "generalization": generalization,
        "contradictions": contradictions,
        "output_dir": str(out.resolve()),
    }


if __name__ == "__main__":
    def progress(step, total, phase_loss, transformer_loss):
        if step == 1 or step % 50 == 0 or step == total:
            print(
                f"{step:4d}/{total} | "
                f"language-phase loss={phase_loss:.5f} | "
                f"transformer loss={transformer_loss:.5f}"
            )

    result = run_language_experiment(progress_callback=progress)
    print("\nResults saved to:", result["output_dir"])
    print(json.dumps(result["report"]["generalization_summary"], indent=2))
