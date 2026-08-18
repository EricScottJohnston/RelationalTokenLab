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

from topology_models import (
    RELATION_NAMES,
    UNKNOWN,
    GraphSample,
    check_coherence_exact,
    collate_samples,
    infer_relation_exact,
    make_dynamic_episode,
    train_baselines,
)


EVENTS = ["BASE", "CUT", "RECONNECT", "CONTRADICTION", "REPAIR"]


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dynamic(
    transformer,
    gnn,
    *,
    node_count: int,
    episodes: int,
    max_nodes: int,
    max_edges: int,
    seed: int,
):
    rng = random.Random(seed)
    device = torch.device("cpu")
    transformer.eval()
    gnn.eval()

    accum = {
        event: {
            "n": 0,
            "exact_relation": 0,
            "transformer_relation": 0,
            "gnn_relation": 0,
            "exact_coherence": 0,
            "transformer_coherence": 0,
            "gnn_coherence": 0,
        }
        for event in EVENTS
    }

    with torch.no_grad():
        for _ in range(episodes):
            episode = make_dynamic_episode(rng, node_count)

            for event, sample in episode:
                batch = collate_samples([sample], max_nodes, max_edges, device)
                t_rel, t_coh = transformer(batch)
                g_rel, g_coh = gnn(batch)

                t_rel_pred = int(t_rel.argmax(dim=1).item())
                g_rel_pred = int(g_rel.argmax(dim=1).item())
                t_coh_pred = int(t_coh.argmax(dim=1).item()) == 1
                g_coh_pred = int(g_coh.argmax(dim=1).item()) == 1

                exact_rel = infer_relation_exact(
                    sample.node_count, sample.edges, sample.src, sample.dst
                )
                exact_coh = check_coherence_exact(sample.node_count, sample.edges)

                expected_rel = sample.relation_target
                expected_coh = sample.coherent

                a = accum[event]
                a["n"] += 1

                # Relation score is meaningful for coherent states. During a contradiction,
                # multiple paths may disagree; we deliberately do not score relation there.
                if expected_coh:
                    a["exact_relation"] += int(exact_rel == expected_rel)
                    a["transformer_relation"] += int(t_rel_pred == expected_rel)
                    a["gnn_relation"] += int(g_rel_pred == expected_rel)

                a["exact_coherence"] += int(exact_coh == expected_coh)
                a["transformer_coherence"] += int(t_coh_pred == expected_coh)
                a["gnn_coherence"] += int(g_coh_pred == expected_coh)

    rows = []
    for event in EVENTS:
        a = accum[event]
        coherent_event = event != "CONTRADICTION"
        denom_rel = a["n"] if coherent_event else 0
        rows.append(
            {
                "node_count": node_count,
                "event": event,
                "episodes": a["n"],
                "exact_relation_accuracy": (
                    a["exact_relation"] / denom_rel if denom_rel else None
                ),
                "transformer_relation_accuracy": (
                    a["transformer_relation"] / denom_rel if denom_rel else None
                ),
                "gnn_relation_accuracy": (
                    a["gnn_relation"] / denom_rel if denom_rel else None
                ),
                "exact_coherence_accuracy": a["exact_coherence"] / a["n"],
                "transformer_coherence_accuracy": a["transformer_coherence"] / a["n"],
                "gnn_coherence_accuracy": a["gnn_coherence"] / a["n"],
            }
        )
    return rows


def aggregate_by_size(rows: List[dict]):
    sizes = sorted({r["node_count"] for r in rows})
    out = []
    for n in sizes:
        rr = [r for r in rows if r["node_count"] == n]

        coherent_rows = [r for r in rr if r["event"] != "CONTRADICTION"]

        def mean_key(key, source_rows):
            vals = [r[key] for r in source_rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        out.append(
            {
                "node_count": n,
                "exact_relation": mean_key("exact_relation_accuracy", coherent_rows),
                "transformer_relation": mean_key("transformer_relation_accuracy", coherent_rows),
                "gnn_relation": mean_key("gnn_relation_accuracy", coherent_rows),
                "exact_coherence": mean_key("exact_coherence_accuracy", rr),
                "transformer_coherence": mean_key("transformer_coherence_accuracy", rr),
                "gnn_coherence": mean_key("gnn_coherence_accuracy", rr),
            }
        )
    return out


def plot_by_size(path: Path, summary: List[dict], metric: str, title: str, ylabel: str, train_max_nodes: int):
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_subplot(111)

    x = [r["node_count"] for r in summary]
    ax.plot(x, [r[f"exact_{metric}"] for r in summary], marker="o", label="Exact relational engine")
    ax.plot(x, [r[f"transformer_{metric}"] for r in summary], marker="o", label="Transformer baseline")
    ax.plot(x, [r[f"gnn_{metric}"] for r in summary], marker="o", label="GNN baseline")
    ax.axvline(train_max_nodes, linestyle="--", label=f"Max training graph size = {train_max_nodes}")

    ax.set_title(title)
    ax.set_xlabel("Nodes in topology-change episode")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_events(path: Path, rows: List[dict], node_count: int):
    rr = [r for r in rows if r["node_count"] == node_count]
    labels = [r["event"] for r in rr]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(10, 5.4))
    ax = fig.add_subplot(111)

    ax.plot(
        x,
        [r["exact_coherence_accuracy"] for r in rr],
        marker="o",
        label="Exact relational engine",
    )
    ax.plot(
        x,
        [r["transformer_coherence_accuracy"] for r in rr],
        marker="o",
        label="Transformer baseline",
    )
    ax.plot(
        x,
        [r["gnn_coherence_accuracy"] for r in rr],
        marker="o",
        label="GNN baseline",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Coherence classification accuracy")
    ax.set_title(f"Changing Relational Boundary — {node_count}-Node Episodes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def detailed_episode_text(
    transformer,
    gnn,
    *,
    node_count: int,
    max_nodes: int,
    max_edges: int,
    seed: int,
) -> str:
    rng = random.Random(seed)
    device = torch.device("cpu")
    episode = make_dynamic_episode(rng, node_count)

    lines = []
    lines.append("EXPERIMENT 2 — ONE CHANGING-TOPOLOGY EPISODE")
    lines.append("=" * 78)
    lines.append(
        "The models are NOT retrained between states. "
        "Only the relation graph changes."
    )
    lines.append("")

    transformer.eval()
    gnn.eval()

    with torch.no_grad():
        for event, sample in episode:
            batch = collate_samples([sample], max_nodes, max_edges, device)
            t_rel, t_coh = transformer(batch)
            g_rel, g_coh = gnn(batch)

            exact_rel = infer_relation_exact(sample.node_count, sample.edges, sample.src, sample.dst)
            exact_coh = check_coherence_exact(sample.node_count, sample.edges)

            t_rel_pred = int(t_rel.argmax(dim=1).item())
            g_rel_pred = int(g_rel.argmax(dim=1).item())
            t_coh_pred = int(t_coh.argmax(dim=1).item()) == 1
            g_coh_pred = int(g_coh.argmax(dim=1).item()) == 1

            lines.append(f"[{event}]")
            lines.append(
                f"  graph: nodes={sample.node_count}, undirected edges={len(sample.edges)}"
            )
            lines.append(f"  query: {sample.src} -> {sample.dst}")
            lines.append(
                f"  expected relation: "
                f"{RELATION_NAMES[sample.relation_target]}"
            )
            lines.append(
                f"  exact relation: {RELATION_NAMES[exact_rel]}"
            )
            lines.append(
                f"  transformer relation: {RELATION_NAMES[t_rel_pred]}"
            )
            lines.append(
                f"  GNN relation: {RELATION_NAMES[g_rel_pred]}"
            )
            lines.append(
                f"  expected coherence={sample.coherent}; "
                f"exact={exact_coh}; transformer={t_coh_pred}; GNN={g_coh_pred}"
            )
            lines.append("")

    return "\n".join(lines)


def run_topology_experiment(
    *,
    output_dir: str | Path = "topology_results",
    seed: int = 19,
    train_min_nodes: int = 4,
    train_max_nodes: int = 8,
    test_sizes=(8, 12, 16, 20, 24),
    max_nodes: int = 24,
    max_edges: int = 40,
    steps: int = 850,
    batch_size: int = 96,
    episodes_per_size: int = 250,
    progress_callback=None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transformer, gnn, history = train_baselines(
        seed=seed,
        train_min_nodes=train_min_nodes,
        train_max_nodes=train_max_nodes,
        test_max_nodes=max_nodes,
        max_edges=max_edges,
        steps=steps,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    all_rows = []
    for idx, n in enumerate(test_sizes):
        rows = evaluate_dynamic(
            transformer,
            gnn,
            node_count=n,
            episodes=episodes_per_size,
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed=seed + 1000 + idx * 97,
        )
        all_rows.extend(rows)

    summary = aggregate_by_size(all_rows)

    save_csv(out / "topology_event_results.csv", all_rows)
    save_csv(out / "topology_size_summary.csv", summary)

    plot_by_size(
        out / "topology_relation_generalization.png",
        summary,
        metric="relation",
        title="Experiment 2: Relation Inference Under Changing Topology",
        ylabel="Relation accuracy on coherent states",
        train_max_nodes=train_max_nodes,
    )
    plot_by_size(
        out / "topology_coherence_generalization.png",
        summary,
        metric="coherence",
        title="Experiment 2: Coherence Detection Under Changing Topology",
        ylabel="Coherence classification accuracy",
        train_max_nodes=train_max_nodes,
    )
    plot_events(
        out / "topology_event_sequence.png",
        all_rows,
        node_count=max(test_sizes),
    )

    episode_text = detailed_episode_text(
        transformer,
        gnn,
        node_count=max(test_sizes),
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed + 7777,
    )
    (out / "topology_episode.txt").write_text(episode_text, encoding="utf-8")

    report = {
        "experiment": "Changing relational topology / boundary",
        "seed": seed,
        "train_node_range": [train_min_nodes, train_max_nodes],
        "test_sizes": list(test_sizes),
        "steps": steps,
        "batch_size": batch_size,
        "episodes_per_size": episodes_per_size,
        "dynamic_sequence": [
            "BASE: coherent tree; query pair connected",
            "CUT: remove a path edge; query becomes disconnected/UNKNOWN",
            "RECONNECT: add a correct cross-component relation",
            "CONTRADICTION: add a wrong edge; graph becomes incoherent",
            "REPAIR: remove wrong edge; coherence restored",
        ],
        "size_summary": summary,
        "interpretation": (
            "This experiment isolates topology from language. Relation IDs 0..3 are "
            "given directly to all systems because Experiment 1 already tested "
            "language-to-relation mapping. The exact relational engine is supplied "
            "the Z4 composition/coherence law by construction. The transformer and "
            "GNN must learn graph inference and coherence from static graphs with "
            "4..8 nodes, then are evaluated without retraining on topology-changing "
            "episodes up to 24 nodes. A positive result demonstrates a structural "
            "advantage for explicit relational computation on this synthetic algebra; "
            "it does not by itself establish superiority on arbitrary real-world graphs."
        ),
    }
    (out / "topology_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "transformer": transformer.state_dict(),
            "gnn": gnn.state_dict(),
            "report": report,
        },
        out / "topology_models.pt",
    )

    return {
        "report": report,
        "event_rows": all_rows,
        "summary": summary,
        "episode_text": episode_text,
        "output_dir": str(out.resolve()),
    }


if __name__ == "__main__":
    def progress(step, total, transformer_loss, gnn_loss):
        if step == 1 or step % 50 == 0 or step == total:
            print(
                f"{step:4d}/{total} | "
                f"transformer loss={transformer_loss:.5f} | "
                f"GNN loss={gnn_loss:.5f}"
            )

    result = run_topology_experiment(progress_callback=progress)
    print("\nResults:", result["output_dir"])
    print(result["episode_text"])
