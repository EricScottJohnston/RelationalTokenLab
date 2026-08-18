"""Experiment 7 — Lexicon Bootstrapping.  Locked before running.

CORE QUESTION
    Experiment 6 showed the reader works when it recognises a relation word and
    fails when it does not, and that composition on top of a correct reading is
    essentially exact (0.998 on chained sentences). Can that exactness be used
    the other way round -- as an equation solved for the words the reader does
    not know?

THE MECHANISM
    Polarity is a Z2 group, so a chain composes by XOR, so a corpus of readable
    sentences is a linear system over GF(2) whose unknowns are the polarities
    of the unreadable phrases. Kind is not a group but is a constraint --
    composition is defined only within a kind -- so kind is settled by which
    system a phrase survives in. Solve, add what was learned back in, solve
    again. See bootstrap_solver.py.

WHAT IS MEASURED
    A  recovery against how much of the lexicon is hidden (10% to 90%). The
       point of the sweep is the threshold: below some number of known anchors
       the system goes underdetermined, and where that sits is the practical
       answer to how large a seed lexicon has to be before it can grow itself.
    B  whether the bootstrap loop earns its place -- does round 2 and beyond
       identify anything round 1 could not.
    C  robustness to a wrong reading, and contradiction detection, which are
       the same mechanism seen from two sides.

CONTROLS
    random       assign each hidden phrase a random kind and polarity
    majority     assign the commonest class among the phrases still known
    scrambled    THE KEY CONTROL. Generate the corpus with polarities drawn at
                 random instead of from a potential. The world stops being
                 consistent, the equations stop having a solution, and recovery
                 must collapse. If it does not, something other than the
                 algebra is doing the work.

WHAT A PASS WOULD MEAN
    That the machine can extend its own lexicon from text it can only partly
    read, using composition as the identification rule, and that it declines to
    guess when the corpus does not pin an answer down.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bootstrap_data import (
    ALL_PHRASES,
    BALANCED_KINDS,
    PHRASE_OWNER,
    build_corpus,
    verify_consistency,
)
from bootstrap_solver import (
    baseline_majority,
    baseline_random,
    bootstrap,
    contradiction_report,
    score,
)

HIDDEN_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
READER_ERRORS = [0.02, 0.05]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_sweep(path, rows):
    fig = plt.figure(figsize=(10.2, 6.0))
    ax = fig.add_subplot(111)
    x = [r["hidden_fraction"] for r in rows]
    ax.plot(x, [r["overall"] for r in rows], marker="o", label="Recovered (kind + polarity)")
    ax.plot(x, [r["identified_fraction"] for r in rows], marker="s",
            label="Claimed at all")
    ax.plot(x, [r["accuracy_among_identified"] for r in rows], marker="^",
            label="Correct when claimed")
    ax.plot(x, [r["random"] for r in rows], linestyle=":", color="gray", label="Random")
    ax.plot(x, [r["majority"] for r in rows], linestyle="--", color="gray", label="Majority class")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x)
    ax.set_xlabel("Fraction of the lexicon hidden from the reader")
    ax.set_ylabel("Fraction of hidden phrases")
    ax.set_title("Experiment 7: recovering unknown relation words by solving the composition")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_rounds(path, rows):
    fig = plt.figure(figsize=(9.4, 5.6))
    ax = fig.add_subplot(111)
    for r in rows:
        per = r["per_round"]
        ax.plot(range(1, len(per) + 1), np.cumsum(per), marker="o",
                label=f"{int(r['hidden_fraction'] * 100)}% hidden")
    ax.set_xlabel("Bootstrap round")
    ax.set_ylabel("Phrases identified (cumulative)")
    ax.set_title("Experiment 7: does feeding what was learned back in identify more")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


LOCKED_CRITERIA_TEXT = {
    "recovery_at_30_above_80": "at 30% hidden, kind and polarity both right on >= 0.80 of hidden phrases",
    "precision_at_30_above_95": "at 30% hidden, >= 0.95 correct among the phrases it claims",
    "recovery_at_50_above_60": "at 50% hidden, >= 0.60 recovered",
    "beats_random_at_30_by_50": "at 30% hidden, recovery minus random >= 0.50",
    "beats_majority_at_30_by_40": "at 30% hidden, recovery minus majority class >= 0.40",
    "scrambled_collapses_below_10": "scrambled corpus at 30% hidden recovers <= 0.10",
    "bootstrap_needs_second_round": "at 50% hidden, round 2 or later identifies something",
    "contradictions_detected_above_80": "at 5% reader error, conflicts found >= 0.80 of those injected",
    "declines_when_starved": "claims a smaller fraction at 90% hidden than at 10% hidden",
}


def locked_criteria(sweep, scrambled, contradiction) -> Dict[str, bool]:
    by_f = {r["hidden_fraction"]: r for r in sweep}
    at30 = by_f.get(0.3, {})
    at50 = by_f.get(0.5, {})
    at10 = by_f.get(0.1, {})
    at90 = by_f.get(0.9, {})
    err5 = next((r for r in contradiction if abs(r["reader_error"] - 0.05) < 1e-9), {})
    per_round_50 = at50.get("per_round", [0])
    return {
        "recovery_at_30_above_80": at30.get("overall", 0.0) >= 0.80,
        "precision_at_30_above_95": at30.get("accuracy_among_identified", 0.0) >= 0.95,
        "recovery_at_50_above_60": at50.get("overall", 0.0) >= 0.60,
        "beats_random_at_30_by_50": at30.get("overall", 0.0) - at30.get("random", 0.0) >= 0.50,
        "beats_majority_at_30_by_40": at30.get("overall", 0.0) - at30.get("majority", 0.0) >= 0.40,
        "scrambled_collapses_below_10": scrambled.get("overall", 1.0) <= 0.10,
        "bootstrap_needs_second_round": sum(per_round_50[1:]) > 0,
        "contradictions_detected_above_80": err5.get("detection_ratio", 0.0) >= 0.80,
        "declines_when_starved":
            at90.get("identified_fraction", 1.0) < at10.get("identified_fraction", 0.0),
    }


def run_experiment(
    *,
    output_dir="bootstrap_results",
    seed=91,
    n_entities=60,
    edges_per_kind=900,
    hidden_fractions=None,
    reader_errors=None,
    event_callback=None,
):
    fractions = list(hidden_fractions or HIDDEN_FRACTIONS)
    errors = list(reader_errors or READER_ERRORS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    def emit(kind, payload):
        if event_callback:
            event_callback(kind, payload)

    emit("status", {
        "text": (f"Lexicon: {len(ALL_PHRASES)} transitive phrases across "
                 f"{len(BALANCED_KINDS)} kinds ({', '.join(BALANCED_KINDS)}). "
                 f"Corpus: {n_entities} entities, {edges_per_kind} sentences per kind. "
                 f"Only transitive relations appear -- a relation that does not compose "
                 f"carries no equation and could never be recovered this way."),
        "overall": 0.02})

    # ---------------- A. hidden-fraction sweep ----------------
    sweep: List[Dict] = []
    for i, f in enumerate(fractions):
        corpus = build_corpus(seed=seed + i, n_entities=n_entities,
                              edges_per_kind=edges_per_kind, hidden_fraction=f)
        res = bootstrap(corpus)
        sc = score(corpus, res)
        row = {
            "hidden_fraction": f,
            **{k: v for k, v in sc.items()},
            "random": baseline_random(corpus, seed + 500 + i)["overall"],
            "majority": baseline_majority(corpus)["overall"],
            "per_round": [r["identified_this_round"] for r in res.rounds],
            "corpus_consistent": all(v["edges_violating_potential"] == 0
                                     for v in verify_consistency(corpus).values()),
            # How often each hidden phrase occurs. A phrase seen once cannot
            # have its kind settled at all, so this bounds what is achievable.
            "median_occurrences": corpus.summary()["occurrences_per_hidden_phrase"]["median"],
            "phrases_seen_3_or_fewer": corpus.summary()["occurrences_per_hidden_phrase"]["at_or_below_3"],
        }
        sweep.append(row)
        emit("sweep_done", {"row": row, "overall": 0.02 + 0.62 * (i + 1) / len(fractions)})

    # ---------------- B. scrambled control ----------------
    scr_corpus = build_corpus(seed=seed + 77, n_entities=n_entities,
                              edges_per_kind=edges_per_kind, hidden_fraction=0.3,
                              consistent=False)
    scr_res = bootstrap(scr_corpus)
    scrambled = {**score(scr_corpus, scr_res),
                 "per_round": [r["identified_this_round"] for r in scr_res.rounds]}
    emit("scrambled_done", {"row": scrambled, "overall": 0.72})

    # ---------------- C. reader error and contradiction ----------------
    contradiction: List[Dict] = []
    for i, err in enumerate(errors):
        corpus = build_corpus(seed=seed + 300 + i, n_entities=n_entities,
                              edges_per_kind=edges_per_kind, hidden_fraction=0.3,
                              reader_error=err)
        res = bootstrap(corpus)
        sc = score(corpus, res)
        cr = contradiction_report(corpus, res)
        row = {"reader_error": err, "overall": sc["overall"],
               "accuracy_among_identified": sc["accuracy_among_identified"],
               "identified_fraction": sc["identified_fraction"], **cr}
        contradiction.append(row)
        emit("contradiction_done", {"row": row,
                                    "overall": 0.74 + 0.20 * (i + 1) / len(errors)})

    # ---------------- Report ----------------
    write_csv(out / "bootstrap_sweep.csv",
              [{k: v for k, v in r.items() if k != "per_round"} for r in sweep])
    write_csv(out / "bootstrap_contradiction.csv", contradiction)
    plot_sweep(out / "bootstrap_sweep.png", sweep)
    plot_rounds(out / "bootstrap_rounds.png", sweep)

    criteria = locked_criteria(sweep, scrambled, contradiction)
    report = {
        "experiment": "Experiment 7 — Lexicon Bootstrapping",
        "locked_before_run": True,
        "hyperparameters": {
            "seed": seed,
            "n_entities": n_entities,
            "edges_per_kind": edges_per_kind,
            "hidden_fractions": fractions,
            "reader_errors": errors,
        },
        "core_question": (
            "Can the composition rule be solved backwards to identify relation words the "
            "reader does not know, and does the result feed back in to identify more?"
        ),
        "mechanism": (
            "Every readable sentence is one equation psi(a) XOR psi(b) = polarity over GF(2); "
            "every unreadable one adds a variable. Potentials are gauge-free and never resolve, "
            "but a phrase's polarity is a difference of potentials and does. Kind is settled by "
            "which of the four systems the phrase survives in."
        ),
        "kinds_used": BALANCED_KINDS,
        "phrases_in_play": len(ALL_PHRASES),
        "sweep": sweep,
        "scrambled_control": scrambled,
        "contradiction": contradiction,
        "locked_success_criteria": LOCKED_CRITERIA_TEXT,
        "criterion_results": criteria,
        "all_locked_criteria_met": all(criteria.values()),
        "interpretation_rule": (
            "Recovery well above both baselines, with the scrambled corpus collapsing, means "
            "the composition algebra is doing the identification and nothing else is. High "
            "accuracy among claimed phrases together with a falling claim rate as the lexicon "
            "is starved means the mechanism knows what it does not know. Recovery that "
            "survives the scrambled control would mean something other than the algebra is "
            "carrying it, and the result would not support the claim. Mixed results stay mixed."
        ),
        "known_limits": (
            "Only transitive relations can be recovered: a relation that does not compose "
            "carries no equation. Phrases appearing in too few sentences, or between entities "
            "the rest of the corpus does not reach, are unidentifiable by construction. "
            "The corpus here is generated, so it is consistent by design; real text is not."
        ),
    }
    (out / "bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # A worked example, so the mechanism is visible rather than asserted.
    demo = build_corpus(seed=seed, n_entities=n_entities,
                        edges_per_kind=edges_per_kind, hidden_fraction=0.3)
    demo_res = bootstrap(demo)
    truth = demo.phrase_truth()
    lines = ["EXPERIMENT 7 — WHAT THE SOLVER SAW AND WHAT IT CONCLUDED", "=" * 64, ""]
    shown = 0
    for p, got in sorted(demo_res.identified.items()):
        if p not in truth or shown >= 10:
            continue
        want = truth[p]
        lines += [f"UNKNOWN PHRASE: '{p}'",
                  f"   solver concluded : {got[0]} / polarity {got[1]}",
                  f"   ground truth     : {want[0]} / polarity {want[1]}  "
                  f"({'correct' if got == want else 'WRONG'})",
                  "   sentences it appeared in:"]
        for e in [e for e in demo.edges if e.phrase == p][:3]:
            lines.append(f"      {e.text}   [{e.source} -> {e.target}]")
        lines.append("")
        shown += 1
    if demo_res.unidentified:
        lines += ["", "NOT CLAIMED (the corpus did not pin these down)", "-" * 40]
        lines += [f"   '{p}'" for p in demo_res.unidentified[:15]]
    (out / "worked_examples.txt").write_text("\n".join(lines), encoding="utf-8")

    emit("done_status", {"text": "Experiment complete.", "overall": 1.0})
    return {"output_dir": str(out.resolve()), "report": report}


if __name__ == "__main__":
    def cb(kind, p):
        if kind in ("status", "done_status"):
            print(p["text"])
        elif kind == "sweep_done":
            r = p["row"]
            print(f"hidden {int(r['hidden_fraction']*100):2d}%: recovered={r['overall']:.3f} "
                  f"claimed={r['identified_fraction']:.3f} "
                  f"correct_when_claimed={r['accuracy_among_identified']:.3f} "
                  f"rounds={r['rounds_used']}")
        elif kind == "scrambled_done":
            print(f"scrambled: recovered={p['row']['overall']:.3f}")
        elif kind == "contradiction_done":
            r = p["row"]
            print(f"reader error {r['reader_error']:.0%}: recovered={r['overall']:.3f} "
                  f"conflicts={r['flagged']}/{r['injected']}")

    result = run_experiment(event_callback=cb)
    print("\nResults:", result["output_dir"])
    print(json.dumps(result["report"]["criterion_results"], indent=2))
