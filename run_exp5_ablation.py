"""Experiment 5 closure ablation.

Runs the locked Experiment 5 protocol twice with identical settings and seed,
changing exactly one thing: whether the system state is quantized onto a
learned finite set after every composition step.

    cleanup_off : continuous state. Reproduces the original run.
    cleanup_on  : closure property restored (Experiments 1-3 lineage).

Locked prediction, recorded before running:
    Compound-intervention complete-delta-signature accuracy was 11.2% in the
    original run. If closure is the bottleneck it should rise materially with
    cleanup on. If it does not, the 64-dimension state capacity is the limit
    instead, and that is the next thing to test.

Existing results in system_grammar_results/ are not touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from system_grammar_experiment import run_experiment

OUT = Path("system_grammar_results_v2")
ARMS = [("cleanup_off", False), ("cleanup_on", True)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, cleanup in ARMS:
        started = time.time()
        print(f"\n=== {name} (cleanup={cleanup}) ===", flush=True)

        def cb(kind, payload, _name=name):
            if kind == "budget_done":
                row = payload["row"]
                print(
                    f"[{_name}] budget {payload['budget']:3d} | "
                    f"transfer={row['transfer_role_exact']:.3f} "
                    f"scratch={row['scratch_role_exact']:.3f} "
                    f"scrambled={row['scrambled_role_exact']:.3f} "
                    f"topo_blind={row['topology_blind_role_exact']:.3f} "
                    f"transformer={row['transformer_role_exact']:.3f}",
                    flush=True,
                )
            elif kind in ("status", "done_status"):
                print(f"[{_name}] {payload['text']}", flush=True)

        result = run_experiment(
            output_dir=str(OUT / name),
            cleanup=cleanup,
            event_callback=cb,
        )
        report = result["report"]
        elapsed = time.time() - started

        summary[name] = {
            "cleanup": cleanup,
            "minutes": round(elapsed / 60, 1),
            "sample_efficiency": report["sample_efficiency"],
            "hard_at_max_budget": report["hard_administrative_metrics_at_max_budget"],
            "compound_at_max_budget": report["compound_intervention_metrics_at_max_budget"],
            "criterion_results": report["criterion_results"],
        }
        print(f"[{name}] done in {elapsed/60:.1f} min", flush=True)

    # Head-to-head on the metric the prediction is about.
    off = summary["cleanup_off"]["compound_at_max_budget"]["transfer"]
    on = summary["cleanup_on"]["compound_at_max_budget"]["transfer"]
    summary["comparison"] = {
        "metric": "compound complete_delta_signature_accuracy (transfer arm)",
        "cleanup_off": off["complete_delta_signature_accuracy"],
        "cleanup_on": on["complete_delta_signature_accuracy"],
        "delta": on["complete_delta_signature_accuracy"] - off["complete_delta_signature_accuracy"],
    }

    (OUT / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== COMPARISON ===")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
