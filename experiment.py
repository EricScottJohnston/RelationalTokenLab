from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import train_models, evaluate_by_length, evaluate_contradictions
from relational_core import topology_change_demo


def save_csv(path: Path, results: Dict[str, list]) -> None:
    keys = list(results.keys())
    rows = zip(*(results[k] for k in keys))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def save_plot(
    path: Path,
    results: Dict[str, list],
    title: str,
    ylabel: str,
    train_max_len: int,
) -> None:
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_subplot(111)

    x = results["length"]
    ax.plot(x, results["exact"], marker="o", markersize=3, label="Exact relational closure")
    ax.plot(x, results["phase"], marker="o", markersize=3, label="Learned phase composition")
    ax.plot(x, results["transformer"], marker="o", markersize=3, label="Tiny transformer")
    ax.axvline(train_max_len, linestyle="--", label=f"Training length limit = {train_max_len}")

    ax.set_title(title)
    ax.set_xlabel("Path / loop length")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(results: Dict[str, list], train_max_len: int) -> dict:
    x = np.array(results["length"])
    mask_in = x <= train_max_len
    mask_out = x > train_max_len

    summary = {}
    for key in ("exact", "phase", "transformer"):
        y = np.array(results[key], dtype=float)
        summary[key] = {
            "mean_in_distribution": float(y[mask_in].mean()) if mask_in.any() else None,
            "mean_out_of_distribution": float(y[mask_out].mean()) if mask_out.any() else None,
            "last_length_accuracy": float(y[-1]),
        }
    return summary


def run_experiment(
    *,
    output_dir: str | Path = "results",
    seed: int = 7,
    train_max_len: int = 5,
    test_max_len: int = 32,
    steps: int = 700,
    batch_size: int = 192,
    examples_per_length: int = 500,
    progress_callback: Optional[Callable] = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase_model, transformer, history = train_models(
        seed=seed,
        train_min_len=1,
        train_max_len=train_max_len,
        test_max_len=test_max_len,
        steps=steps,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    generalization = evaluate_by_length(
        phase_model,
        transformer,
        min_len=1,
        max_len=test_max_len,
        examples_per_length=examples_per_length,
        seed=seed + 1000,
    )
    contradictions = evaluate_contradictions(
        phase_model,
        transformer,
        min_len=3,
        max_len=test_max_len,
        examples_per_length=examples_per_length,
        seed=seed + 2000,
    )

    save_csv(out / "generalization.csv", generalization)
    save_csv(out / "contradictions.csv", contradictions)

    save_plot(
        out / "generalization.png",
        generalization,
        "Relational Composition: Generalization Beyond Training Length",
        "Exact answer accuracy",
        train_max_len,
    )
    save_plot(
        out / "contradictions.png",
        contradictions,
        "Loop Closure: Contradiction Detection",
        "Consistent / inconsistent classification accuracy",
        train_max_len,
    )

    topo = topology_change_demo(seed=seed)
    (out / "topology_demo.txt").write_text(topo, encoding="utf-8")

    learned_angles = phase_model.angles.detach().cpu().numpy().tolist()
    report = {
        "seed": seed,
        "train_max_len": train_max_len,
        "test_max_len": test_max_len,
        "steps": steps,
        "batch_size": batch_size,
        "examples_per_length": examples_per_length,
        "learned_phase_angles_radians_raw": learned_angles,
        "generalization_summary": summarize(generalization, train_max_len),
        "contradiction_summary": summarize(contradictions, train_max_len),
        "interpretation_warning": (
            "This is a synthetic inductive-bias experiment. The exact relational "
            "engine is given the correct Z4 composition law, while the transformer "
            "must learn the task. A result favoring the relational models does not "
            "by itself establish superiority on natural language or prove a new "
            "general AI architecture."
        ),
    }

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    torch.save(
        {
            "phase_model": phase_model.state_dict(),
            "transformer": transformer.state_dict(),
            "config": report,
        },
        out / "models.pt",
    )

    return {
        "report": report,
        "generalization": generalization,
        "contradictions": contradictions,
        "topology_demo": topo,
        "output_dir": str(out.resolve()),
    }


if __name__ == "__main__":
    def progress(step, total, phase_loss, trans_loss):
        if step == 1 or step % 50 == 0 or step == total:
            print(
                f"{step:4d}/{total} | "
                f"phase loss={phase_loss:.5f} | transformer loss={trans_loss:.5f}"
            )

    result = run_experiment(progress_callback=progress)
    print()
    print(result["topology_demo"])
    print()
    print("Results saved to:", result["output_dir"])
    print(json.dumps(result["report"]["generalization_summary"], indent=2))
