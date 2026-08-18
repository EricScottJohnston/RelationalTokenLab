"""Experiment 4 closure ablation.

Runs the locked Experiment 4 protocol twice with identical settings and seed,
changing exactly one thing: whether the relational state is quantized onto a
learned finite set after every composition step.

    cleanup_off : continuous state. Reproduces the original run.
    cleanup_on  : closure property restored (Experiments 1-3 lineage).

Locked predictions, recorded before running:
    1. Absolute hard-test accuracy at 128 legal examples was 42.75%. It should
       rise with cleanup on, because depth-4 chains are where a continuous
       state accumulates error.
    2. The transfer-minus-scratch gap should NOT change. Negative transfer is
       about the operator carrying wrong content, not about lost precision.
       If the gap moves, the negative-transfer reading was wrong.

Existing results in crossdomain_results/ are not touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from crossdomain_experiment import run_crossdomain_experiment

OUT = Path("crossdomain_results_v2")
ARMS = [("cleanup_off", False), ("cleanup_on", True)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, cleanup in ARMS:
        started = time.time()
        print(f"\n=== {name} (cleanup={cleanup}) ===", flush=True)

        def cb(kind, payload, _name=name):
            if kind == "budget_done":
                r = payload.get("row", {})
                print(
                    f"[{_name}] budget {r.get('legal_examples'):3d} | "
                    f"transfer={r.get('transfer_accuracy', 0):.3f} "
                    f"scratch={r.get('scratch_accuracy', 0):.3f} "
                    f"scrambled={r.get('scrambled_accuracy', 0):.3f} "
                    f"transformer={r.get('transformer_accuracy', 0):.3f}",
                    flush=True,
                )
            elif kind in ("status", "done_status"):
                print(f"[{_name}] {payload['text']}", flush=True)

        result = run_crossdomain_experiment(
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
            "special_tests_at_max_budget": report["special_tests_at_max_budget"],
            "criterion_results": report["criterion_results"],
        }
        print(f"[{name}] done in {elapsed/60:.1f} min", flush=True)

    def at(arm, n):
        for row in summary[arm]["sample_efficiency"]:
            if row["legal_examples"] == n:
                return row
        return {}

    off128, on128 = at("cleanup_off", 128), at("cleanup_on", 128)
    off64, on64 = at("cleanup_off", 64), at("cleanup_on", 64)
    summary["comparison"] = {
        "prediction_1_absolute_accuracy_at_128": {
            "cleanup_off": off128.get("transfer_accuracy"),
            "cleanup_on": on128.get("transfer_accuracy"),
        },
        "prediction_2_transfer_minus_scratch_at_64": {
            "cleanup_off": round(off64.get("transfer_accuracy", 0) - off64.get("scratch_accuracy", 0), 4),
            "cleanup_on": round(on64.get("transfer_accuracy", 0) - on64.get("scratch_accuracy", 0), 4),
        },
    }

    (OUT / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== COMPARISON ===")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
