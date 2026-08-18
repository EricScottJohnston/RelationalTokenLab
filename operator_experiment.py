from __future__ import annotations

from pathlib import Path
import csv
import json
import random
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_models import (
    UNKNOWN,
    exact_detect_coherence,
    exact_infer_state,
    evaluate_depth,
    learned_table_report,
    make_dynamic_episode,
    model_detect_coherence,
    model_infer_state,
    train_models,
)


def save_dict_csv(path: Path, data: dict):
    keys = list(data.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(data[k] for k in keys)))


def plot_depth(path: Path, data: dict, train_max_len: int):
    fig = plt.figure(figsize=(9.5, 5.6))
    ax = fig.add_subplot(111)
    x = data["length"]

    ax.plot(x, data["exact"], marker="o", markersize=2, label="Exact hidden law")
    ax.plot(x, data["structured"], marker="o", markersize=2, label="Learned operator + structure")
    ax.plot(x, data["unconstrained"], marker="o", markersize=2, label="Learned operator, unconstrained")
    ax.plot(x, data["transformer"], marker="o", markersize=2, label="Tiny transformer")
    ax.axvline(train_max_len, linestyle="--", label=f"Training depth limit = {train_max_len}")

    ax.set_title("Experiment 3: Can the Model Learn the Composition Law?")
    ax.set_xlabel("Composition depth")
    ax.set_ylabel("Exact relation accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_depth(data: dict, train_max_len: int):
    x = np.array(data["length"])
    out = {}
    for key in ("exact", "structured", "unconstrained", "transformer"):
        y = np.array(data[key], dtype=float)
        in_mask = x <= train_max_len
        out_mask = x > train_max_len
        out[key] = {
            "mean_in_distribution": float(y[in_mask].mean()),
            "mean_out_of_distribution": float(y[out_mask].mean()),
            "length_32_accuracy": float(y[x == 32][0]) if 32 in x else None,
            "last_length_accuracy": float(y[-1]),
        }
    return out


def evaluate_topology(
    structured,
    unconstrained,
    *,
    sizes=(8, 12, 16, 20, 24),
    episodes_per_size=180,
    seed=3001,
    progress_callback=None,
):
    rng = random.Random(seed)
    rows = []

    total = len(sizes) * episodes_per_size
    done = 0

    for n in sizes:
        counters = {
            "exact_relation": 0,
            "structured_relation": 0,
            "unconstrained_relation": 0,
            "relation_total": 0,
            "exact_coherence": 0,
            "structured_coherence": 0,
            "unconstrained_coherence": 0,
            "coherence_total": 0,
        }

        for _ in range(episodes_per_size):
            episode = make_dynamic_episode(rng, n)

            for state in episode:
                # Score relation only when the graph is coherent. The contradiction
                # state can contain path-dependent answers by construction.
                if state.coherent:
                    truth_rel = state.target_relation
                    counters["relation_total"] += 1
                    counters["exact_relation"] += int(exact_infer_state(state) == truth_rel)
                    counters["structured_relation"] += int(
                        model_infer_state(structured, state) == truth_rel
                    )
                    counters["unconstrained_relation"] += int(
                        model_infer_state(unconstrained, state) == truth_rel
                    )

                counters["coherence_total"] += 1
                counters["exact_coherence"] += int(
                    exact_detect_coherence(state) == state.coherent
                )
                counters["structured_coherence"] += int(
                    model_detect_coherence(structured, state) == state.coherent
                )
                counters["unconstrained_coherence"] += int(
                    model_detect_coherence(unconstrained, state) == state.coherent
                )

            done += 1
            if progress_callback and (
                done == 1 or done % 25 == 0 or done == total
            ):
                progress_callback(done, total, n)

        rows.append(
            {
                "node_count": n,
                "exact_relation": counters["exact_relation"] / counters["relation_total"],
                "structured_relation": counters["structured_relation"] / counters["relation_total"],
                "unconstrained_relation": counters["unconstrained_relation"] / counters["relation_total"],
                "exact_coherence": counters["exact_coherence"] / counters["coherence_total"],
                "structured_coherence": counters["structured_coherence"] / counters["coherence_total"],
                "unconstrained_coherence": counters["unconstrained_coherence"] / counters["coherence_total"],
            }
        )

    return rows


def save_rows_csv(path: Path, rows: List[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_topology(path: Path, rows: List[dict], metric: str, title: str):
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_subplot(111)

    x = [r["node_count"] for r in rows]
    ax.plot(x, [r[f"exact_{metric}"] for r in rows], marker="o", label="Exact hidden law")
    ax.plot(x, [r[f"structured_{metric}"] for r in rows], marker="o", label="Learned operator + structure")
    ax.plot(x, [r[f"unconstrained_{metric}"] for r in rows], marker="o", label="Learned operator, unconstrained")
    ax.axvline(8, linestyle="--", label="Largest graph size analogous to prior training regime")

    ax.set_title(title)
    ax.set_xlabel("Nodes in changing-topology episode")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_table(table):
    lines = []
    lines.append("      b=0  b=1  b=2  b=3")
    for a, row in enumerate(table):
        lines.append(f"a={a}   " + "    ".join(str(x) for x in row))
    return "\n".join(lines)


def run_operator_experiment(
    *,
    output_dir: str | Path = "operator_results",
    seed: int = 29,
    train_max_len: int = 5,
    test_max_len: int = 64,
    steps: int = 950,
    batch_size: int = 160,
    examples_per_length: int = 400,
    topology_episodes_per_size: int = 180,
    event_callback=None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def train_progress(step, total, s_task, s_total, u_loss, t_loss, regs):
        if event_callback:
            event_callback(
                "train",
                {
                    "step": step,
                    "total": total,
                    "structured_task_loss": s_task,
                    "structured_total_loss": s_total,
                    "unconstrained_loss": u_loss,
                    "transformer_loss": t_loss,
                    "regs": regs,
                    "overall_pct": 0.70 * step / total,
                },
            )

    structured, unconstrained, transformer = train_models(
        seed=seed,
        train_max_len=train_max_len,
        test_max_len=test_max_len,
        steps=steps,
        batch_size=batch_size,
        progress_callback=train_progress,
    )

    if event_callback:
        event_callback("status", {"text": "Training finished. Evaluating unseen composition depths..."})

    def depth_progress(done, total):
        if event_callback:
            event_callback(
                "depth",
                {
                    "done": done,
                    "total": total,
                    "overall_pct": 0.70 + 0.15 * done / total,
                },
            )

    depth = evaluate_depth(
        structured,
        unconstrained,
        transformer,
        max_len=test_max_len,
        examples_per_length=examples_per_length,
        seed=seed + 1000,
        progress_callback=depth_progress,
    )

    if event_callback:
        event_callback("status", {"text": "Depth evaluation finished. Testing changing topologies..."})

    def topo_progress(done, total, n):
        if event_callback:
            event_callback(
                "topology",
                {
                    "done": done,
                    "total": total,
                    "node_count": n,
                    "overall_pct": 0.85 + 0.15 * done / total,
                },
            )

    topology = evaluate_topology(
        structured,
        unconstrained,
        episodes_per_size=topology_episodes_per_size,
        seed=seed + 2000,
        progress_callback=topo_progress,
    )

    structured_table = learned_table_report(structured)
    unconstrained_table = learned_table_report(unconstrained)

    save_dict_csv(out / "operator_depth_generalization.csv", depth)
    save_rows_csv(out / "operator_topology_summary.csv", topology)

    plot_depth(out / "operator_depth_generalization.png", depth, train_max_len)
    plot_topology(
        out / "operator_topology_relation.png",
        topology,
        "relation",
        "Experiment 3: Learned Law Under Changing Topology — Relation Inference",
    )
    plot_topology(
        out / "operator_topology_coherence.png",
        topology,
        "coherence",
        "Experiment 3: Learned Law Under Changing Topology — Coherence",
    )

    table_text = (
        "STRUCTURED LEARNED OPERATOR\n"
        "===========================\n"
        + format_table(structured_table["predicted_cayley_table"])
        + "\n\nHIDDEN GROUND-TRUTH TABLE (used only by generator/evaluator)\n"
        + format_table(structured_table["ground_truth_hidden_table"])
        + "\n\nUNCONSTRAINED LEARNED OPERATOR\n"
        "==============================\n"
        + format_table(unconstrained_table["predicted_cayley_table"])
        + "\n"
    )
    (out / "learned_operator_tables.txt").write_text(table_text, encoding="utf-8")

    report = {
        "experiment": "Learn the relational composition law rather than supplying it",
        "seed": seed,
        "train_max_len": train_max_len,
        "test_max_len": test_max_len,
        "steps": steps,
        "batch_size": batch_size,
        "examples_per_length": examples_per_length,
        "topology_episodes_per_size": topology_episodes_per_size,
        "critical_design_fact": (
            "The two learned-operator models contain NO complex multiplication and NO "
            "modular-addition composition rule. Their binary operator is an MLP. "
            "Ground-truth Z4 composition is used only to generate target labels and evaluate results."
        ),
        "structured_priors": [
            "relation 0 designated as identity",
            "inverse pairing supplied: 0<->0, 1<->3, 2<->2",
            "associativity regularizer",
            "closure regularizer: pair outputs should land near some learned relation state, without specifying which",
            "prototype separation regularizer",
        ],
        "depth_summary": summarize_depth(depth, train_max_len),
        "structured_operator": structured_table,
        "unconstrained_operator": unconstrained_table,
        "topology_summary": topology,
        "interpretation_warning": (
            "This is still a synthetic four-relation world. The structured model is "
            "not given the hidden composition table, but it is given strong algebraic "
            "priors including identity, inverse pairing, associativity, and closure. "
            "A successful result would show that these constraints plus shallow task "
            "examples are sufficient to learn a reusable composition operator in this "
            "world. It would not establish that arbitrary real-world relational laws "
            "can be discovered automatically."
        ),
    }

    (out / "operator_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "structured": structured.state_dict(),
            "unconstrained": unconstrained.state_dict(),
            "transformer": transformer.state_dict(),
            "report": report,
        },
        out / "operator_models.pt",
    )

    if event_callback:
        event_callback("status", {"text": "All evaluation finished. Writing result files complete."})

    return {
        "report": report,
        "output_dir": str(out.resolve()),
        "table_text": table_text,
    }


if __name__ == "__main__":
    def cb(kind, payload):
        if kind == "train":
            if payload["step"] == 1 or payload["step"] % 50 == 0 or payload["step"] == payload["total"]:
                print(
                    f"{payload['step']:4d}/{payload['total']} | "
                    f"structured task={payload['structured_task_loss']:.5f} | "
                    f"unconstrained={payload['unconstrained_loss']:.5f} | "
                    f"transformer={payload['transformer_loss']:.5f}"
                )
        elif kind == "status":
            print(payload["text"])
        elif kind == "depth":
            if payload["done"] % 8 == 0 or payload["done"] == payload["total"]:
                print(f"Depth evaluation: {payload['done']}/{payload['total']}")
        elif kind == "topology":
            if payload["done"] % 50 == 0 or payload["done"] == payload["total"]:
                print(
                    f"Topology evaluation: {payload['done']}/{payload['total']} "
                    f"(currently {payload['node_count']} nodes)"
                )

    result = run_operator_experiment(event_callback=cb)
    print("\nResults:", result["output_dir"])
    print(result["table_text"])
