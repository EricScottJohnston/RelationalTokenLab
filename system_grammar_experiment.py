from __future__ import annotations

from pathlib import Path
import csv
import json
import random
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from system_grammar_data import (
    Vocabulary,
    generate_dataset,
)
from system_grammar_models import (
    SystemGrammarModel,
    TinySystemTransformer,
    evaluate_metrics,
    parameter_count,
    predict_system,
    predict_transformer,
    train_system_model,
    train_transformer,
)

DEFAULT_BUDGETS = [8, 16, 32, 64, 128]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_transfer(path, rows):
    fig = plt.figure(figsize=(9.6, 5.8))
    ax = fig.add_subplot(111)
    x = [r["examples"] for r in rows]
    for key, label in [
        ("transfer", "Transferred system grammar"),
        ("scratch", "Scratch system model"),
        ("scrambled", "Role-scrambled transfer"),
        ("topology_blind", "Topology-blind transfer"),
        ("transformer", "Tiny transformer"),
    ]:
        ax.plot(x, [r[f"{key}_role_exact"] for r in rows], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, labels=[str(v) for v in x])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Labeled administrative examples")
    ax.set_ylabel("Exact causal-delta signature accuracy")
    ax.set_title("Experiment 5: Cross-Domain System-Grammar Transfer")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_metrics(path, metrics_by_model):
    names = ["transfer", "scratch", "scrambled", "topology_blind", "transformer"]
    metric_keys = [
        "system_role_exact_accuracy",
        "topology_vs_geometry_accuracy",
        "affected_set_micro_f1",
        "invariant_set_precision",
        "counterfactual_direction_accuracy",
    ]
    labels = ["Role exact", "Topology vs geometry", "Affected F1", "Invariant precision", "Direction"]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    x = np.arange(len(metric_keys))
    width = 0.15
    offsets = np.linspace(-2, 2, len(names)) * width
    for off, name in zip(offsets, names):
        ax.bar(x + off, [metrics_by_model[name][k] for k in metric_keys], width, label=name)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 5: Hard Administrative Causal Reasoning")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_compound(path, compound_metrics):
    names = ["transfer", "scratch", "scrambled", "topology_blind", "transformer"]
    vals = [compound_metrics[n]["complete_delta_signature_accuracy"] for n in names]
    fig = plt.figure(figsize=(9, 5.3))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(names)), vals)
    ax.set_xticks(np.arange(len(names)), ["Transfer", "Scratch", "Scrambled", "Topo-blind", "Transformer"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Exact complete causal-delta signature")
    ax.set_title("Experiment 5: Unseen Compound Interventions")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def locked_criteria(rows, hard, compound):
    at32 = {r["examples"]: r for r in rows}[32]
    c = {
        "transfer_minus_scratch_at_32":
            at32["transfer_role_exact"] - at32["scratch_role_exact"] >= 0.15,
        "transfer_minus_transformer_at_32":
            at32["transfer_role_exact"] - at32["transformer_role_exact"] >= 0.10,
        "hard_system_role_identification":
            hard["transfer"]["system_role_exact_accuracy"] >= 0.90,
        "hard_topology_vs_geometry":
            hard["transfer"]["topology_vs_geometry_accuracy"] >= 0.95,
        "hard_affected_set_f1":
            hard["transfer"]["affected_set_micro_f1"] >= 0.90,
        "hard_invariant_set_precision":
            hard["transfer"]["invariant_set_precision"] >= 0.95,
        "hard_counterfactual_direction":
            hard["transfer"]["counterfactual_direction_accuracy"] >= 0.90,
        "compound_complete_delta_signature":
            compound["transfer"]["complete_delta_signature_accuracy"] >= 0.85,
        "role_scramble_materially_worse":
            hard["transfer"]["system_role_exact_accuracy"] - hard["scrambled"]["system_role_exact_accuracy"] >= 0.10,
    }
    return c


def run_experiment(
    *,
    output_dir="system_grammar_results",
    seed=53,
    mechanical_steps=1100,
    admin_steps=350,
    batch_size=72,
    budgets=None,
    event_callback=None,
    cleanup=False,
    num_states=512,
):
    """cleanup=False reproduces the original continuous-state run exactly.

    cleanup=True quantizes the system state onto a learned finite set after
    every composition step (the closure property from Experiments 1-3).
    """
    if budgets is None:
        budgets = list(DEFAULT_BUDGETS)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --------------------------------------------------
    # DOMAIN A: physical fluid-control systems
    # --------------------------------------------------
    mech_train = generate_dataset(
        seed + 1, 7000, "mechanical",
        compound_sizes=(1, 1, 1, 2),
        observation_probability=0.15,
        policy_structural_fraction=0.30,
    )
    mech_eval = generate_dataset(
        seed + 2, 1400, "mechanical",
        compound_sizes=(1, 2),
        observation_probability=0.18,
        policy_structural_fraction=0.30,
    )
    mech_vocab = Vocabulary(mech_train + mech_eval)

    base = SystemGrammarModel(len(mech_vocab.itos), cleanup=cleanup, num_states=num_states)

    def mech_prog(step, total, loss, exact):
        if event_callback:
            event_callback("mechanical_train", {
                "step": step, "total": total, "loss": loss, "exact": exact,
                "overall": 0.20 * step / total,
            })

    train_system_model(
        base, mech_train, mech_vocab,
        steps=mechanical_steps, batch_size=batch_size, lr=2.2e-3,
        seed=seed + 3, progress=mech_prog,
    )
    pm, ym = predict_system(base, mech_eval, mech_vocab)
    mechanical_metrics = evaluate_metrics(pm, ym)

    if event_callback:
        event_callback("status", {
            "text": f"Mechanical pretraining complete. Role-exact={mechanical_metrics['system_role_exact_accuracy']:.3f}"
        })

    # --------------------------------------------------
    # DOMAIN B: administrative/legal institutional systems
    # Few-shot training only uses SINGLE interventions.
    # Hard test uses different mixes, observations, and policy structural cases.
    # Compound test uses 2-3 simultaneous interventions.
    # --------------------------------------------------
    admin_pool = generate_dataset(
        seed + 10, max(budgets), "administrative",
        compound_sizes=(1,),
        observation_probability=0.15,
        policy_structural_fraction=0.35,
    )
    hard_test = generate_dataset(
        seed + 11, 1500, "administrative",
        compound_sizes=(1,),
        observation_probability=0.25,
        policy_structural_fraction=0.45,
    )
    compound_test = generate_dataset(
        seed + 12, 1200, "administrative",
        compound_sizes=(2, 3),
        observation_probability=0.10,
        policy_structural_fraction=0.0,
    )

    # Build vocabulary from all surface forms, but no labels leak from test.
    admin_vocab = Vocabulary(admin_pool + hard_test + compound_test)

    rows = []
    final_models = {}
    total_runs = len(budgets) * 5
    completed = 0

    for budget in budgets:
        subset = admin_pool[:budget]

        # 1. Intact transferred system grammar. Only legal/admin encoder trains.
        transfer = base.transfer_shell(len(admin_vocab.itos))
        def p_transfer(step, total, loss, exact):
            if event_callback and (step == 1 or step % 50 == 0 or step == total):
                event_callback("admin_train", {
                    "budget": budget, "model": "transfer",
                    "step": step, "total": total, "loss": loss, "exact": exact,
                    "overall": 0.20 + 0.80 * ((completed + step/total) / total_runs),
                })
        train_system_model(
            transfer, subset, admin_vocab,
            steps=admin_steps, batch_size=min(batch_size, max(16, budget)),
            lr=3e-3, seed=seed + 100 + budget, progress=p_transfer,
        )
        completed += 1

        # 2. Scratch system model.
        scratch = SystemGrammarModel(len(admin_vocab.itos), cleanup=cleanup, num_states=num_states)
        def p_scratch(step, total, loss, exact):
            if event_callback and (step == 1 or step % 50 == 0 or step == total):
                event_callback("admin_train", {
                    "budget": budget, "model": "scratch",
                    "step": step, "total": total, "loss": loss, "exact": exact,
                    "overall": 0.20 + 0.80 * ((completed + step/total) / total_runs),
                })
        train_system_model(
            scratch, subset, admin_vocab,
            steps=admin_steps, batch_size=min(batch_size, max(16, budget)),
            lr=2.5e-3, seed=seed + 200 + budget, progress=p_scratch,
        )
        completed += 1

        # 3. Role-scrambled transferred core.
        scrambled = base.role_scrambled_shell(len(admin_vocab.itos), seed + 300 + budget)
        def p_scr(step, total, loss, exact):
            if event_callback and (step == 1 or step % 50 == 0 or step == total):
                event_callback("admin_train", {
                    "budget": budget, "model": "scrambled",
                    "step": step, "total": total, "loss": loss, "exact": exact,
                    "overall": 0.20 + 0.80 * ((completed + step/total) / total_runs),
                })
        train_system_model(
            scrambled, subset, admin_vocab,
            steps=admin_steps, batch_size=min(batch_size, max(16, budget)),
            lr=3e-3, seed=seed + 400 + budget, progress=p_scr,
        )
        completed += 1

        # 4. Topology-blind ablation.
        blind = base.topology_blind_shell(len(admin_vocab.itos))
        def p_blind(step, total, loss, exact):
            if event_callback and (step == 1 or step % 50 == 0 or step == total):
                event_callback("admin_train", {
                    "budget": budget, "model": "topology_blind",
                    "step": step, "total": total, "loss": loss, "exact": exact,
                    "overall": 0.20 + 0.80 * ((completed + step/total) / total_runs),
                })
        train_system_model(
            blind, subset, admin_vocab,
            steps=admin_steps, batch_size=min(batch_size, max(16, budget)),
            lr=3e-3, seed=seed + 500 + budget, progress=p_blind,
        )
        completed += 1

        # 5. Transformer baseline.
        transformer = TinySystemTransformer(len(admin_vocab.itos))
        def p_tf(step, total, loss, exact):
            if event_callback and (step == 1 or step % 50 == 0 or step == total):
                event_callback("admin_train", {
                    "budget": budget, "model": "transformer",
                    "step": step, "total": total, "loss": loss, "exact": exact,
                    "overall": 0.20 + 0.80 * ((completed + step/total) / total_runs),
                })
        train_transformer(
            transformer, subset, admin_vocab,
            steps=admin_steps, batch_size=min(batch_size, max(16, budget)),
            lr=2e-3, seed=seed + 600 + budget, progress=p_tf,
        )
        completed += 1

        ph, yh = predict_system(transfer, hard_test, admin_vocab)
        sh, _ = predict_system(scratch, hard_test, admin_vocab)
        rh, _ = predict_system(scrambled, hard_test, admin_vocab)
        bh, _ = predict_system(blind, hard_test, admin_vocab)
        th, _ = predict_transformer(transformer, hard_test, admin_vocab)

        mt = evaluate_metrics(ph, yh)
        ms = evaluate_metrics(sh, yh)
        mr = evaluate_metrics(rh, yh)
        mb = evaluate_metrics(bh, yh)
        mf = evaluate_metrics(th, yh)

        rows.append({
            "examples": budget,
            "transfer_role_exact": mt["system_role_exact_accuracy"],
            "scratch_role_exact": ms["system_role_exact_accuracy"],
            "scrambled_role_exact": mr["system_role_exact_accuracy"],
            "topology_blind_role_exact": mb["system_role_exact_accuracy"],
            "transformer_role_exact": mf["system_role_exact_accuracy"],
        })

        if event_callback:
            event_callback("budget_done", {
                "budget": budget, "row": rows[-1],
                "overall": 0.20 + 0.80 * completed / total_runs,
            })

        if budget == max(budgets):
            final_models = {
                "transfer": transfer,
                "scratch": scratch,
                "scrambled": scrambled,
                "topology_blind": blind,
                "transformer": transformer,
            }

    # Final detailed hard test.
    hard_metrics = {}
    compound_metrics = {}
    for name, model in final_models.items():
        if name == "transformer":
            p, y = predict_transformer(model, hard_test, admin_vocab)
            pc, yc = predict_transformer(model, compound_test, admin_vocab)
        else:
            p, y = predict_system(model, hard_test, admin_vocab)
            pc, yc = predict_system(model, compound_test, admin_vocab)
        hard_metrics[name] = evaluate_metrics(p, y)
        compound_metrics[name] = evaluate_metrics(pc, yc)

    criteria = locked_criteria(rows, hard_metrics, compound_metrics)

    write_csv(out / "system_grammar_sample_efficiency.csv", rows)
    plot_transfer(out / "system_grammar_sample_efficiency.png", rows)
    plot_metrics(out / "system_grammar_hard_metrics.png", hard_metrics)
    plot_compound(out / "system_grammar_compound_interventions.png", compound_metrics)

    report = {
        "experiment": "Experiment 5 — System Grammar and Causal Transfer",
        "seed": seed,
        "locked_before_run": True,
        "state_cleanup": cleanup,
        "num_relation_states": (num_states if cleanup else None),
        "cleanup_note": (
            "State is quantized onto a learned finite set after every composition step "
            "(closure property from Experiments 1-3)." if cleanup else
            "Continuous state, no closure step. Reproduces the original Experiment 5 run."
        ),
        "system_ontology": ["Boundary", "Policy", "Topology", "Geometry", "Level", "Rate", "Constraint", "Delay", "Information"],
        "behavior_definition": "Behavior is the trajectory generated by the system; it is not treated as a primitive role.",
        "causal_definition": (
            "Given a system and a deliberate intervention, identify which role(s) changed, "
            "which downstream roles can change through the learned system grammar, which "
            "roles remain invariant, and the qualitative direction of the nominated terminal level."
        ),
        "domain_A": "Physical fluid-control systems with tanks, pipes, pumps, valves, sensors, limits, and controllers.",
        "domain_B": "Administrative/legal institutional systems with jurisdiction, authority/referral links, review-channel capacity, queues, processing rates, constraints, waiting periods, and reporting.",
        "transfer_rule": (
            "Only the learned system core and system-output heads transfer. They are frozen. "
            "The administrative domain receives a fresh vocabulary and sentence encoder."
        ),
        "anti_cheating_design": [
            "Mechanical and administrative domains use separate vocabularies and fresh domain encoders.",
            "Transferred system core and output heads are frozen during administrative training.",
            "Scratch system model gets the identical labeled administrative examples.",
            "Role-scrambled transfer damages recurrent hidden-state semantics while preserving architecture and parameter scale.",
            "Topology-blind ablation collapses topology/geometry distinction in the frozen head.",
            "Transformer receives identical labeled administrative examples.",
            "Administrative few-shot training contains only single interventions.",
            "Compound test contains 2-3 simultaneous interventions never seen during administrative training.",
            "Observations are mixed with interventions; observations must not be treated as causal manipulations.",
        ],
        "training": {
            "mechanical_steps": mechanical_steps,
            "administrative_steps_per_model_per_budget": admin_steps,
            "batch_size": batch_size,
            "budgets": budgets,
            "mechanical_train_examples": len(mech_train),
            "mechanical_eval_examples": len(mech_eval),
            "administrative_hard_test_examples": len(hard_test),
            "compound_test_examples": len(compound_test),
        },
        "mechanical_pretraining_metrics": mechanical_metrics,
        "sample_efficiency": rows,
        "hard_administrative_metrics_at_max_budget": hard_metrics,
        "compound_intervention_metrics_at_max_budget": compound_metrics,
        "locked_success_criteria": {
            "at_32_examples_transfer_minus_scratch": ">= 0.15 exact delta signature",
            "at_32_examples_transfer_minus_transformer": ">= 0.10 exact delta signature",
            "hard_system_role_identification": ">= 0.90 exact delta signature",
            "hard_topology_vs_geometry": ">= 0.95",
            "hard_affected_set_f1": ">= 0.90",
            "hard_invariant_set_precision": ">= 0.95",
            "hard_counterfactual_direction": ">= 0.90",
            "compound_complete_delta_signature": ">= 0.85",
            "role_scramble_materially_worse": "transfer >= scrambled + 0.10 on exact role signature",
        },
        "criterion_results": criteria,
        "all_locked_criteria_met": all(criteria.values()),
        "parameter_counts_at_max_budget": {
            name: {
                "total": parameter_count(model),
                "trainable": parameter_count(model, trainable_only=True),
            }
            for name, model in final_models.items()
        },
        "interpretation_rule": (
            "Do not collapse a mixed result to a single boolean. The central positive-transfer claim requires "
            "the intact transferred system grammar to show few-shot advantage and to retain topology/geometry, "
            "affected/invariant-set, and counterfactual reasoning. Failure of one subcriterion should be reported "
            "as a specific boundary rather than erasing successful subresults."
        ),
    }

    (out / "system_grammar_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable samples.
    lines = ["EXPERIMENT 5 SAMPLE CASES", "=========================", ""]
    lines += ["MECHANICAL DOMAIN", "-----------------"]
    for c in mech_eval[:4]:
        lines.append(" | ".join(c.facts))
        lines.append(c.event_text)
        lines.append(f"delta={c.delta} affected={c.affected} tg={c.topology_geometry} direction={c.direction}")
        lines.append("")
    lines += ["ADMINISTRATIVE DOMAIN", "---------------------"]
    for c in hard_test[:4]:
        lines.append(" | ".join(c.facts))
        lines.append(c.event_text)
        lines.append(f"delta={c.delta} affected={c.affected} tg={c.topology_geometry} direction={c.direction}")
        lines.append("")
    (out / "sample_cases.txt").write_text("\n".join(lines), encoding="utf-8")

    torch.save({
        "mechanical_base": base.state_dict(),
        "final_models": {name: model.state_dict() for name, model in final_models.items()},
        "report": report,
    }, out / "system_grammar_models.pt")

    if event_callback:
        event_callback("done_status", {"text": "Experiment complete.", "overall": 1.0})

    return {"output_dir": str(out.resolve()), "report": report}


if __name__ == "__main__":
    def cb(kind, p):
        if kind == "mechanical_train":
            print(f"Mechanical {p['step']:4d}/{p['total']} loss={p['loss']:.4f} exact={p['exact']:.3f}")
        elif kind == "admin_train":
            print(f"Admin n={p['budget']:3d} {p['model']:14s} {p['step']:3d}/{p['total']} loss={p['loss']:.4f} exact={p['exact']:.3f}")
        elif kind == "budget_done":
            print("Budget:", p["row"])
        elif kind in ("status", "done_status"):
            print(p["text"])
    r = run_experiment(event_callback=cb)
    print("Results:", r["output_dir"])
    print("Criteria:", r["report"]["criterion_results"])
