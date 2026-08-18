"""Experiment 6 — sentence-level relation induction.  Locked before running.

CORE QUESTION
    Experiment 1b showed that a relation between two *word forms* is
    recoverable from surface alone and then composes by addition past training
    depth. Does the same hold one level up, between two *entities in a
    sentence*?

WHAT IS HELD OUT
    Version 1 of this experiment held out whole sentence shapes: it trained on
    "{a} {phrase} {b}" and tested on passives, constructions and noun forms the
    model had never encountered. That is not a generalization test. Three tiers
    scored below the majority-class floor, which is what an impossible task
    looks like.

    Here every shape appears in training and the *wording* is held out. See
    sentence_data.py for the full split table.

THE TIERS
    A  familiar        trained phrase and frame, new entities  (sanity)
    B  new frame       held-out plain frame, trained phrase
    C  inflection      "will prevent" for a relation whose modal and
                       progressive frames were held out
    D  synonym         held-out phrase sharing no stem  (ARBITRARINESS
                       CONTROL -- expected to sit at the floor)
    E  construction    held-out construction, no relation word anywhere
                       (THIS IS THE EXPERIMENT)
    F  passive         passive for a relation whose passives were held out
    G  nominal         held-out noun-form phrasing

CONTROLS
    majority     always predict the most frequent training class
    transformer  larger model, identical data, no explicit relation states
    shuffled     tier E with the words in random order. Syntax destroyed,
                 vocabulary kept. If accuracy survives, the model is doing
                 bag-of-words and the structural claim is dead.
    span-swap    tier A with the two entity markers exchanged. The model is
                 now being asked about the reverse pair, so accuracy against
                 the unchanged label must fall. If it does not, the span
                 markers are being ignored and no direction is being read.

COMPOSITION
    Two sentences joined at a shared entity, read independently, polarities
    XOR'd. Nothing about chaining is trained. This is the part that tests the
    topology claim rather than the reading claim.
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
import torch

from relational_lexicon import ALL_RELATIONS, KINDS, stats as lexicon_stats
from sentence_data import (
    CharVocabulary,
    SentenceGenerator,
    TIER_DESCRIPTIONS,
    TIER_NAMES,
    audit,
    shuffle_words,
    swap_spans,
)
from sentence_models import (
    SentenceResolver,
    TinySentenceTransformer,
    evaluate,
    evaluate_chains,
    majority_floor,
    parameter_count,
    train_model,
)

CHAIN_TIERS = ["A_familiar", "E_construction"]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_tiers(path, rows):
    fig = plt.figure(figsize=(11.5, 6.2))
    ax = fig.add_subplot(111)
    x = np.arange(len(rows))
    w = 0.28
    ax.bar(x - w, [r["resolver"] for r in rows], w, label="Relational resolver")
    ax.bar(x, [r["transformer"] for r in rows], w, label="Tiny transformer")
    ax.bar(x + w, [r["majority"] for r in rows], w, color="#999999", label="Majority class")
    ax.set_xticks(x)
    ax.set_xticklabels([r["tier"].split("_", 1)[0] + "\n" + r["tier"].split("_", 1)[1]
                        for r in rows], fontsize=9)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Kind and polarity both correct")
    ax.set_title("Experiment 6: relation recovered from a sentence, by what was held out")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_controls(path, controls):
    fig = plt.figure(figsize=(9.0, 5.4))
    ax = fig.add_subplot(111)
    labels = list(controls)
    ax.bar(labels, [controls[k] for k in labels], color="#5577aa")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Kind and polarity both correct")
    ax.set_title("Experiment 6: controls")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


LOCKED_CRITERIA_TEXT = {
    "A_familiar_above_85": "tier A >= 0.85 (sanity; nothing below reads without it)",
    "B_new_frame_above_80": "tier B >= 0.80",
    "C_inflection_beats_floor_by_30": "tier C minus its floor >= 0.30",
    "E_construction_beats_floor_by_25": "tier E minus its floor >= 0.25",
    "E_construction_beats_transformer_by_10": "tier E minus transformer >= 0.10",
    "F_passive_beats_floor_by_25": "tier F minus its floor >= 0.25",
    "G_nominal_beats_floor_by_20": "tier G minus its floor >= 0.20",
    "shuffled_E_drops_by_20": "tier E minus word-shuffled tier E >= 0.20",
    "span_swap_A_drops_by_30": "tier A minus span-swapped tier A >= 0.30",
    "chain_E_polarity_beats_50": "composed polarity on unseen constructions >= 0.50 above chance (>= 0.75)",
}


def locked_criteria(tier_rows, controls, chain_rows) -> Dict[str, bool]:
    t = {r["tier"]: r for r in tier_rows}
    c = {r["tier"]: r for r in chain_rows}

    def acc(name):
        return t.get(name, {}).get("resolver", 0.0)

    def floor(name):
        return t.get(name, {}).get("majority", 0.0)

    return {
        "A_familiar_above_85": acc("A_familiar") >= 0.85,
        "B_new_frame_above_80": acc("B_new_frame") >= 0.80,
        "C_inflection_beats_floor_by_30": acc("C_inflection") - floor("C_inflection") >= 0.30,
        "E_construction_beats_floor_by_25": acc("E_construction") - floor("E_construction") >= 0.25,
        "E_construction_beats_transformer_by_10":
            acc("E_construction") - t.get("E_construction", {}).get("transformer", 0.0) >= 0.10,
        "F_passive_beats_floor_by_25": acc("F_passive") - floor("F_passive") >= 0.25,
        "G_nominal_beats_floor_by_20": acc("G_nominal") - floor("G_nominal") >= 0.20,
        "shuffled_E_drops_by_20":
            acc("E_construction") - controls.get("resolver_shuffled_E", 1.0) >= 0.20,
        "span_swap_A_drops_by_30":
            acc("A_familiar") - controls.get("resolver_spanswap_A", 1.0) >= 0.30,
        "chain_E_polarity_beats_50":
            c.get("E_construction", {}).get("resolver_polarity", 0.0) >= 0.75,
    }


def run_experiment(
    *,
    output_dir="sentence_results",
    seed=71,
    steps=1200,
    batch_size=64,
    lr=2.0e-3,
    n_angles=12,
    train_size=9000,
    test_size=600,
    chain_size=400,
    phrase_holdout=0.25,
    event_callback=None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def emit(kind, payload):
        if event_callback:
            event_callback(kind, payload)

    # ---------------- Data ----------------
    gen = SentenceGenerator(seed=seed, phrase_holdout=phrase_holdout)
    train = gen.training_set(seed + 1, train_size)
    tiers = {name: gen.tier(seed + 10 + i, test_size, name)
             for i, name in enumerate(TIER_NAMES)}
    chains = {name: gen.chains(seed + 50 + i, chain_size, name)
              for i, name in enumerate(CHAIN_TIERS)}

    ctl_rng = random.Random(seed + 900)
    shuffled_E = [shuffle_words(s, ctl_rng) for s in tiers["E_construction"]]
    spanswap_A = [swap_spans(s) for s in tiers["A_familiar"]]

    everything = (train + [s for ds in tiers.values() for s in ds]
                  + shuffled_E + spanswap_A
                  + [c.first for cs in chains.values() for c in cs]
                  + [c.second for cs in chains.values() for c in cs])
    vocab = CharVocabulary(everything)

    data_audit = audit(gen, seed=seed + 777, n=min(test_size, 400))
    emit("status", {
        "text": (f"Training set: {len(train)} sentences across every shape. "
                 f"Test tiers: {len(TIER_NAMES)} x {test_size}. "
                 f"Character vocabulary: {len(vocab.itos)}. "
                 f"Lexicon: {lexicon_stats()['relation_types']} relations, "
                 f"{lexicon_stats()['total_phrases']} phrases."),
        "overall": 0.02})

    resolver = SentenceResolver(len(vocab.itos), n_angles=n_angles)
    transformer = TinySentenceTransformer(len(vocab.itos))

    # ---------------- Training ----------------
    train_model(resolver, train, vocab, steps, batch_size, lr, seed + 2,
                progress=lambda s, t, l, a: emit("train", {
                    "model": "resolver", "step": s, "total": t, "loss": l, "acc": a,
                    "overall": 0.02 + 0.42 * s / t}))
    train_model(transformer, train, vocab, steps, batch_size, lr, seed + 3,
                progress=lambda s, t, l, a: emit("train", {
                    "model": "transformer", "step": s, "total": t, "loss": l, "acc": a,
                    "overall": 0.44 + 0.34 * s / t}))

    torch.save({"state_dict": resolver.state_dict(), "vocab": vocab.itos,
                "n_angles": n_angles, "seed": seed},
               out / "resolver.pt")
    torch.save({"state_dict": transformer.state_dict(), "vocab": vocab.itos,
                "seed": seed},
               out / "transformer.pt")

    # ---------------- Tiers ----------------
    tier_rows: List[Dict] = []
    for i, name in enumerate(TIER_NAMES):
        ds = tiers[name]
        r = evaluate(resolver, ds, vocab)
        tf = evaluate(transformer, ds, vocab)
        row = {
            "tier": name,
            "description": TIER_DESCRIPTIONS[name],
            "resolver": r["both"],
            "resolver_kind": r["kind"],
            "resolver_polarity": r["polarity"],
            "transformer": tf["both"],
            "transformer_kind": tf["kind"],
            "transformer_polarity": tf["polarity"],
            "majority": majority_floor(train, ds),
            "n": r["n"],
        }
        tier_rows.append(row)
        emit("tier_done", {"row": row, "overall": 0.78 + 0.12 * (i + 1) / len(TIER_NAMES)})

    # ---------------- Controls ----------------
    controls = {
        "resolver_shuffled_E": evaluate(resolver, shuffled_E, vocab)["both"],
        "transformer_shuffled_E": evaluate(transformer, shuffled_E, vocab)["both"],
        "resolver_spanswap_A": evaluate(resolver, spanswap_A, vocab)["both"],
        "transformer_spanswap_A": evaluate(transformer, spanswap_A, vocab)["both"],
    }
    emit("controls_done", {"controls": controls, "overall": 0.93})

    # ---------------- Composition ----------------
    chain_rows: List[Dict] = []
    for i, name in enumerate(CHAIN_TIERS):
        cs = chains[name]
        r = evaluate_chains(resolver, cs, vocab)
        tf = evaluate_chains(transformer, cs, vocab)
        row = {
            "tier": name,
            "resolver_polarity": r["polarity"],
            "resolver_kind": r["kind"],
            "resolver_both": r["both"],
            "transformer_polarity": tf["polarity"],
            "transformer_kind": tf["kind"],
            "transformer_both": tf["both"],
            "n": r["n"],
        }
        chain_rows.append(row)
        emit("chain_done", {"row": row, "overall": 0.94 + 0.05 * (i + 1) / len(CHAIN_TIERS)})

    # ---------------- Report ----------------
    write_csv(out / "sentence_tiers.csv", tier_rows)
    write_csv(out / "sentence_chains.csv", chain_rows)
    plot_tiers(out / "sentence_tiers.png", tier_rows)
    plot_controls(out / "sentence_controls.png", controls)

    criteria = locked_criteria(tier_rows, controls, chain_rows)
    by_tier = {r["tier"]: r for r in tier_rows}
    d = by_tier.get("D_synonym", {})
    report = {
        "experiment": "Experiment 6 — Sentence-Level Relation Induction (design 2)",
        "locked_before_run": True,
        "hyperparameters": {
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "n_angles": n_angles,
            "train_size": train_size,
            "test_size_per_tier": test_size,
            "chain_size": chain_size,
            "phrase_holdout_fraction": phrase_holdout,
        },
        "core_question": (
            "Is a relation between two entities in a sentence recoverable from surface "
            "structure, and does it then compose by addition across sentences?"
        ),
        "what_changed_from_design_1": (
            "Design 1 trained on one sentence shape and tested on four unseen shapes, so "
            "four of five tiers were unlearnable and three fell below the majority floor. "
            "Design 2 puts every shape in training and holds out the wording instead."
        ),
        "kinds": KINDS,
        "relation_count": len(ALL_RELATIONS),
        "lexicon": lexicon_stats(),
        "split_report": gen.split_report(),
        "data_audit": data_audit,
        "tiers": tier_rows,
        "controls": controls,
        "composition": chain_rows,
        "parameter_counts": {
            "resolver": parameter_count(resolver),
            "transformer": parameter_count(transformer),
        },
        "locked_success_criteria": LOCKED_CRITERIA_TEXT,
        "criterion_results": criteria,
        "all_locked_criteria_met": all(criteria.values()),
        "arbitrariness_control": {
            "tier": "D_synonym",
            "resolver": d.get("resolver"),
            "majority": d.get("majority"),
            "prediction": (
                "D should sit at its floor. The link between a word's form and its meaning "
                "is arbitrary, so nothing in the characters of 'blocks' identifies it as the "
                "same relation as 'prevents'. D scoring well above floor would mean something "
                "is leaking -- a shared stem, or a frame that gives the relation away -- and "
                "the split needs re-auditing before any other tier is believed."
            ),
            "sits_at_floor": (d.get("resolver", 1.0) - d.get("majority", 0.0)) < 0.15,
        },
        "interpretation_rule": (
            "Tier E is the experiment. Passing E above the floor and above the transformer "
            "means the relation was read out of sentence structure with no relation word "
            "present. Passing the composition check on E means the sentence-level relations "
            "then add the way the word-level ones did. Failing E while passing A and B means "
            "the model is looking words up, which is a lookup table with extra steps. "
            "Mixed results stay mixed."
        ),
    }
    (out / "sentence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["EXPERIMENT 6 SAMPLE SENTENCES", "=" * 60, "", "TRAINING", "-" * 30]
    lines += [s.describe() for s in train[:12]]
    for name in TIER_NAMES:
        lines += ["", f"{name}  —  {TIER_DESCRIPTIONS[name]}", "-" * 30]
        lines += [s.describe() for s in tiers[name][:6]]
    lines += ["", "COMPOSITION CHAINS (unseen constructions)", "-" * 30]
    for c in chains["E_construction"][:6]:
        lines += [f"  1: {c.first.text}   [{c.first.source} -> {c.first.target}]",
                  f"  2: {c.second.text}   [{c.second.source} -> {c.second.target}]",
                  f"     composed: kind={KINDS[c.kind]} polarity={c.polarity}", ""]
    (out / "sample_sentences.txt").write_text("\n".join(lines), encoding="utf-8")

    emit("done_status", {"text": "Experiment complete.", "overall": 1.0})
    return {"output_dir": str(out.resolve()), "report": report}


if __name__ == "__main__":
    def cb(kind, p):
        if kind in ("status", "done_status"):
            print(p["text"])
        elif kind == "tier_done":
            r = p["row"]
            print(f"{r['tier']:16s} resolver={r['resolver']:.3f} "
                  f"transformer={r['transformer']:.3f} floor={r['majority']:.3f}")
        elif kind == "chain_done":
            r = p["row"]
            print(f"chain {r['tier']:14s} polarity={r['resolver_polarity']:.3f} "
                  f"kind={r['resolver_kind']:.3f}")

    result = run_experiment(event_callback=cb)
    print("\nResults:", result["output_dir"])
    print(json.dumps(result["report"]["criterion_results"], indent=2))
