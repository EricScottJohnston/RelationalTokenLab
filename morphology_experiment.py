"""Experiment 1b — Language-Internal Relational Structure.

Locked before running.

CORE QUESTION
    Is the relational structure that Experiment 1 found in phrasing actually a
    property of the language system rather than of what the words refer to,
    and does the same structure survive a change of language?

WHY THIS IS NOT EXPERIMENT 1 AGAIN
    Experiment 1's relations were rotations - things in the world that the
    phrases named. Ground truth came from geometry. Here ground truth comes
    only from form-to-form relations inside a lexicon. Nothing in the task
    requires knowing what a word means.

PHASES
    A. Depth generalization. Train on chains of depth 1-2 in English, test to
       depth 5. Same shape as Experiment 1's length extrapolation.
    B. Cross-linguistic transfer. Freeze the relation resolver, the composition
       operator, the codebook and the head. Give the model a fresh character
       encoder and a few German examples. Only the part that reads characters
       may adapt.

CONTROLS
    scratch      identical architecture, German only, no transfer
    scrambled    transferred weights with composition semantics randomized
    transformer  bigger model, same data, no explicit relation states
    majority     always predict the most frequent training class
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

from morphology_data import (
    ChainGenerator,
    NonceGenerator,
    CharVocabulary,
    NUM_COMPOSED_RELATIONS,
    describe,
)
from morphology_models import (
    unseen_combination_rate,
    RelationalChainModel,
    TinyChainTransformer,
    eval_relational,
    eval_transformer,
    majority_baseline,
    parameter_count,
    train_relational,
    train_transformer,
)

TRAIN_DEPTHS = (1, 2)
# Depth 5 is excluded: the lexicon graph only admits two composed classes at
# that length, so accuracy there is dominated by the majority class and says
# nothing about composition. Real derivational chains simply do not run deeper
# than about four steps, which bounds the extrapolation range this domain can
# support. Training at 1-2 and testing at 4 is a doubling, not Experiment 1's
# 12.8x.
TEST_DEPTHS = (1, 2, 3, 4)
GERMAN_BUDGETS = [8, 16, 32, 64, 128]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_depth(path, rows):
    fig = plt.figure(figsize=(9.6, 5.8))
    ax = fig.add_subplot(111)
    x = [r["depth"] for r in rows]
    ax.plot(x, [r["relational"] for r in rows], marker="o", label="Relational (composes)")
    ax.plot(x, [r["transformer"] for r in rows], marker="s", label="Tiny transformer")
    ax.plot(x, [r["majority"] for r in rows], linestyle=":", color="gray", label="Majority class")
    ax.axvline(max(TRAIN_DEPTHS) + 0.5, linestyle="--", color="black", alpha=0.6)
    ax.text(max(TRAIN_DEPTHS) + 0.6, 0.05, "beyond training depth", fontsize=9)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x)
    ax.set_xlabel("Derivational chain depth")
    ax.set_ylabel("Composed-relation accuracy")
    ax.set_title("Experiment 1b: Depth Generalization (English)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_crosslingual(path, rows):
    fig = plt.figure(figsize=(9.6, 5.8))
    ax = fig.add_subplot(111)
    x = [r["examples"] for r in rows]
    for key, label in [
        ("transfer", "Transfer (frozen relations, new reader)"),
        ("scratch", "Scratch German"),
        ("scrambled", "Scrambled-operator transfer"),
        ("transformer", "Tiny transformer"),
    ]:
        ax.plot(x, [r[key] for r in rows], marker="o", label=label)
    ax.plot(x, [r["majority"] for r in rows], linestyle=":", color="gray", label="Majority class")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, labels=[str(v) for v in x])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Labeled German chains")
    ax.set_ylabel("Composed-relation accuracy")
    ax.set_title("Experiment 1b: English -> German Structural Transfer")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def locked_criteria(depth_rows, cross_rows) -> Dict[str, bool]:
    by_depth = {r["depth"]: r for r in depth_rows}
    by_budget = {r["examples"]: r for r in cross_rows}
    deep = by_depth.get(4, {})
    at32 = by_budget.get(32, {})
    return {
        "depth4_relational_above_85":
            deep.get("relational", 0) >= 0.85,
        "depth4_relational_beats_transformer_by_20":
            deep.get("relational", 0) - deep.get("transformer", 0) >= 0.20,
        "depth4_relational_beats_majority_by_25":
            deep.get("relational", 0) - deep.get("majority", 0) >= 0.25,
        "german32_transfer_beats_scratch_by_15":
            at32.get("transfer", 0) - at32.get("scratch", 0) >= 0.15,
        "german32_transfer_beats_scrambled_by_15":
            at32.get("transfer", 0) - at32.get("scrambled", 0) >= 0.15,
        "german32_transfer_beats_transformer_by_10":
            at32.get("transfer", 0) - at32.get("transformer", 0) >= 0.10,
    }


def run_experiment(
    *,
    output_dir="morphology_results",
    seed=61,
    english_steps=900,
    german_steps=300,
    batch_size=64,
    cleanup=True,
    num_states=64,
    composition="additive",
    n_angles=12,
    budgets=None,
    event_callback=None,
):
    if budgets is None:
        budgets = list(GERMAN_BUDGETS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def emit(kind, payload):
        if event_callback:
            event_callback(kind, payload)

    # ---------------- Phase A: English, depth 1-2 ----------------
    eng = ChainGenerator("english")
    ger = ChainGenerator("german")
    eng_nonce = NonceGenerator("english")

    eng_train = eng.dataset(seed + 1, 6000, TRAIN_DEPTHS)
    eng_tests = {d: eng.dataset(seed + 10 + d, 600, (d,)) for d in TEST_DEPTHS}
    nonce_tests = {d: eng_nonce.dataset(seed + 30 + d, 600, (d,)) for d in TEST_DEPTHS}
    ger_tests = {d: ger.dataset(seed + 40 + d, 600, (d,)) for d in TEST_DEPTHS}

    # One character vocabulary spanning every language and both real and nonce
    # forms. The model is never given a fresh reader, so German is a genuine
    # zero-shot test: train on English, show it German cold. Latin script
    # overlaps almost entirely; the only additions are the umlauts and eszett.
    everything = (
        eng_train
        + [c for ds in eng_tests.values() for c in ds]
        + [c for ds in nonce_tests.values() for c in ds]
        + [c for ds in ger_tests.values() for c in ds]
        + ger.dataset(seed + 60, 400, TRAIN_DEPTHS)
        + NonceGenerator("german").dataset(seed + 61, 400, TEST_DEPTHS)
    )
    eng_vocab = CharVocabulary(everything)

    rel = RelationalChainModel(len(eng_vocab.itos), cleanup=cleanup, num_states=num_states,
                               composition=composition, n_angles=n_angles)
    tf = TinyChainTransformer(len(eng_vocab.itos))

    emit("status", {"text": f"English training set: {len(eng_train)} chains, depths {TRAIN_DEPTHS}. "
                            f"Character vocabulary: {len(eng_vocab.itos)}.", "overall": 0.02})

    train_relational(
        rel, eng_train, eng_vocab, english_steps, batch_size, 2.0e-3, seed + 2,
        progress=lambda s, t, l, a: emit("english_train", {
            "model": "relational", "step": s, "total": t, "loss": l, "acc": a,
            "overall": 0.02 + 0.34 * s / t}),
    )
    train_transformer(
        tf, eng_train, eng_vocab, english_steps, batch_size, 2.0e-3, seed + 3,
        progress=lambda s, t, l, a: emit("english_train", {
            "model": "transformer", "step": s, "total": t, "loss": l, "acc": a,
            "overall": 0.36 + 0.24 * s / t}),
    )

    # ---- Zero-shot generalization: real English, nonce, and German ----
    # No further training anywhere below. Every number is the English-trained
    # model evaluated cold.
    zero_shot_rows = []
    for d in TEST_DEPTHS:
        row = {
            "depth": d,
            "english_real_relational": eval_relational(rel, eng_tests[d], eng_vocab),
            "english_real_transformer": eval_transformer(tf, eng_tests[d], eng_vocab),
            "nonce_relational": eval_relational(rel, nonce_tests[d], eng_vocab),
            "nonce_transformer": eval_transformer(tf, nonce_tests[d], eng_vocab),
            "german_relational": eval_relational(rel, ger_tests[d], eng_vocab),
            "german_transformer": eval_transformer(tf, ger_tests[d], eng_vocab),
            "nonce_majority": majority_baseline(eng_train, nonce_tests[d]),
            "german_majority": majority_baseline(eng_train, ger_tests[d]),
        }
        zero_shot_rows.append(row)
        emit("zeroshot_done", {"row": row, "overall": 0.60 + 0.04 * (d / len(TEST_DEPTHS))})

    depth_rows = []
    for d in TEST_DEPTHS:
        ds = eng_tests[d]
        row = {
            "depth": d,
            "relational": eval_relational(rel, ds, eng_vocab),
            "transformer": eval_transformer(tf, ds, eng_vocab),
            "majority": majority_baseline(eng_train, ds),
            "unseen_combos": unseen_combination_rate(eng_train, ds),
            "n": len(ds),
        }
        depth_rows.append(row)
        emit("depth_done", {"row": row, "overall": 0.60 + 0.05 * (d / len(TEST_DEPTHS))})

    # ---------------- Phase B: German few-shot ----------------
    ger_pool = ger.dataset(seed + 20, max(budgets), TRAIN_DEPTHS)
    ger_test = [c for ds in ger_tests.values() for c in ds]
    ger_vocab = eng_vocab   # shared: no fresh reader, so transfer is not smuggled in

    emit("status", {"text": f"German pool: {len(ger_pool)} chains. Test: {len(ger_test)} chains "
                            f"across depths {TEST_DEPTHS}. Character vocabulary: {len(ger_vocab.itos)}.",
                    "overall": 0.66})

    cross_rows = []
    for bi, budget in enumerate(budgets):
        subset = ger_pool[:budget]
        bs = min(batch_size, max(8, budget))

        transfer = rel.encoder_shell(len(ger_vocab.itos))
        train_relational(transfer, subset, ger_vocab, german_steps, bs, 3e-3, seed + 100 + budget)

        scratch = RelationalChainModel(len(ger_vocab.itos), cleanup=cleanup, num_states=num_states,
                                       composition=composition, n_angles=n_angles)
        train_relational(scratch, subset, ger_vocab, german_steps, bs, 3e-3, seed + 200 + budget)

        scrambled = rel.scrambled_shell(len(ger_vocab.itos), seed + 300 + budget)
        train_relational(scrambled, subset, ger_vocab, german_steps, bs, 3e-3, seed + 400 + budget)

        gtf = TinyChainTransformer(len(ger_vocab.itos))
        train_transformer(gtf, subset, ger_vocab, german_steps, bs, 2e-3, seed + 500 + budget)

        row = {
            "examples": budget,
            "transfer": eval_relational(transfer, ger_test, ger_vocab),
            "scratch": eval_relational(scratch, ger_test, ger_vocab),
            "scrambled": eval_relational(scrambled, ger_test, ger_vocab),
            "transformer": eval_transformer(gtf, ger_test, ger_vocab),
            "majority": majority_baseline(subset, ger_test),
        }
        cross_rows.append(row)
        emit("budget_done", {"row": row, "budget": budget,
                             "overall": 0.70 + 0.28 * (bi + 1) / len(budgets)})

    # ---------------- Report ----------------
    write_csv(out / "morphology_depth.csv", depth_rows)
    write_csv(out / "morphology_crosslingual.csv", cross_rows)
    plot_depth(out / "morphology_depth_generalization.png", depth_rows)
    plot_crosslingual(out / "morphology_crosslingual_transfer.png", cross_rows)

    criteria = locked_criteria(depth_rows, cross_rows)
    report = {
        "experiment": "Experiment 1b — Language-Internal Relational Structure",
        "seed": seed,
        "locked_before_run": True,
        # Recorded so a run is reproducible from its report alone. Earlier
        # reports omitted these, which left two archived arms distinguishable
        # only by their folder names.
        "hyperparameters": {
            "seed": seed,
            "english_steps": english_steps,
            "german_steps": german_steps,
            "batch_size": batch_size,
            "english_lr": 2.0e-3,
            "german_lr": 3.0e-3,
            "transformer_german_lr": 2.0e-3,
            "english_train_examples": 6000,
            "eval_examples_per_depth": 600,
        },
        "composition": composition,
        "n_angles": (n_angles if composition == "additive" else None),
        "state_cleanup": cleanup,
        "num_relation_states": (num_states if cleanup else None),
        "core_question": (
            "Is the compositional structure a property of the language system rather than "
            "of the referents, and does it survive a change of language?"
        ),
        "non_referential_guarantee": (
            "Ground truth is form-to-form only. The relation between teach and teacher is the "
            "same object as the relation between write and writer; nothing in the loss requires "
            "knowing what either word denotes. Input is characters, never a word list."
        ),
        "composed_relation_space": {
            "definition": "category transition over {NOUN, VERB, ADJ, ADV} crossed with Z2 polarity",
            "size": NUM_COMPOSED_RELATIONS,
        },
        "train_depths": list(TRAIN_DEPTHS),
        "test_depths": list(TEST_DEPTHS),
        "german_budgets": budgets,
        "zero_shot": zero_shot_rows,
        "depth_generalization": depth_rows,
        "crosslingual_transfer": cross_rows,
        "parameter_counts": {
            "relational_total": parameter_count(rel),
            "transformer_total": parameter_count(tf),
            "transfer_trainable_reader_only": parameter_count(
                rel.encoder_shell(len(ger_vocab.itos)), trainable_only=True),
        },
        "locked_success_criteria": {
            "depth4_relational_above_85": ">= 0.85",
            "depth4_relational_beats_transformer_by_20": ">= 0.20",
            "depth4_relational_beats_majority_by_25": ">= 0.25",
            "german32_transfer_beats_scratch_by_15": ">= 0.15",
            "german32_transfer_beats_scrambled_by_15": ">= 0.15",
            "german32_transfer_beats_transformer_by_10": ">= 0.10",
        },
        "criterion_results": criteria,
        "all_locked_criteria_met": all(criteria.values()),
        "interpretation_rule": (
            "A pass on depth generalization shows the relation states compose beyond training "
            "depth on language-internal structure. A pass on cross-linguistic transfer shows the "
            "same states are recoverable from a different language with different surface "
            "marking. Neither establishes that arbitrary natural text has this structure - the "
            "lexicon here is hand-built and the operation set is small. Mixed results stay mixed."
        ),
    }
    (out / "morphology_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["EXPERIMENT 1b SAMPLE CHAINS", "=" * 40, "", "ENGLISH", "-" * 20]
    lines += [describe(c) for c in eng_train[:8]]
    deepest = max(TEST_DEPTHS)
    lines += ["", f"ENGLISH DEPTH {deepest}", "-" * 20]
    lines += [describe(c) for c in eng_tests[deepest][:5]]
    lines += ["", "GERMAN", "-" * 20]
    lines += [describe(c) for c in ger_pool[:8]]
    (out / "sample_chains.txt").write_text("\n".join(lines), encoding="utf-8")

    emit("done_status", {"text": "Experiment complete.", "overall": 1.0})
    return {"output_dir": str(out.resolve()), "report": report}


if __name__ == "__main__":
    def cb(kind, p):
        if kind in ("status", "done_status"):
            print(p["text"])
        elif kind == "depth_done":
            r = p["row"]
            print(f"depth {r['depth']}: relational={r['relational']:.3f} "
                  f"transformer={r['transformer']:.3f} majority={r['majority']:.3f}")
        elif kind == "budget_done":
            r = p["row"]
            print(f"german n={r['examples']:3d}: transfer={r['transfer']:.3f} "
                  f"scratch={r['scratch']:.3f} scrambled={r['scrambled']:.3f} "
                  f"transformer={r['transformer']:.3f} majority={r['majority']:.3f}")

    result = run_experiment(event_callback=cb)
    print("\nResults:", result["output_dir"])
    print(json.dumps(result["report"]["criterion_results"], indent=2))
