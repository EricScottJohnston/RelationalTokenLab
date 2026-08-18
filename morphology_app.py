from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from morphology_experiment import run_experiment

LOCK_TEXT = """EXPERIMENT 1b — LANGUAGE-INTERNAL RELATIONAL STRUCTURE
Locked before the run.

CORE QUESTION
Experiment 1 showed that many English phrasings collapse onto four relation
states that then compose past training depth. But those relations were
rotations — things in the world that the phrases named. Ground truth came from
geometry, so the structure could have been living in the referents rather than
in the language.

This experiment removes the referents. Ground truth is form-to-form only.

  nation -> national -> nationalize -> nationalization

The relation between "teach" and "teacher" is the same object as the relation
between "write" and "writer". That identity is a fact about the token system.
Nothing in the task requires knowing what a nation is, and the model never sees
a word list — input is characters.

THE COMPOSED RELATION
The productive derivational operations all advance the same three-cycle:

  NOUN --ADJECTIVAL--> ADJ --CAUSATIVE--> VERB --NOMINALIZE--> NOUN

so a four-step chain returns to NOUN having advanced by three. PLURAL and PAST
are the identity on this cycle. NEGATION is an involution on a separate Z2:
un-un-X == X.

That gives a closed abelian group of order six — Z3 x Z2. Closure is what makes
depth extrapolation measurable, and it is the same property that made Z4 work
in Experiment 1.

WHERE THE SURFACE/STRUCTURE SPLIT LIVES
The same relation surfaces completely differently depending on the stem:
  PLURAL   -s / -es / -en / vowel change / mouse-mice / zero
  PAST     -ed / ablaut / go-went
  NEGATION un- / im- / il- / ir- / in-
Suppletive pairs share no characters at all, so only structural position
identifies the relation.

PHASE A — DEPTH GENERALIZATION (ENGLISH)
Train on chains of depth 1-2. Test at depths 1, 2, 3, 4.
Depth 5 is excluded: the lexicon admits only two composed classes at that
length, so accuracy would be majority-class noise. Real derivational chains do
not run much past four steps. This is a doubling of depth, not Experiment 1's
12.8x.

PHASE B — CROSS-LINGUISTIC TRANSFER (ENGLISH -> GERMAN)
Freeze the relation resolver, the composition operator, the codebook and the
output head. Give the model a fresh character encoder and a few labeled German
chains. Only the part that reads characters may adapt.

German marks these relations differently, and in two cases not at all:
  ADJ -> ADV     English suffixes -ly.  German does nothing:  schnell/schnell
  PLURAL         German zero-plural:    Lehrer/Lehrer
A relation with no surface signal is the sharpest test in the set.

CONTROLS
  scratch      identical architecture, German only, no transfer
  scrambled    transferred weights, composition semantics randomized. The
               output layer is re-randomized rather than permuted, because a
               permutation is something a trainable head learns around for
               free — the flaw in Experiment 4's control.
  transformer  larger model (104k vs 55k params), same data, no explicit
               relation states
  majority     always predict the most frequent training class

LOCKED SUCCESS CRITERIA
At depth 4:
  relational accuracy                      >= 0.85
  relational minus transformer             >= 0.20
  relational minus majority class          >= 0.25
At 32 labeled German chains:
  transfer minus scratch                   >= 0.15
  transfer minus scrambled                 >= 0.15
  transfer minus transformer               >= 0.10

INTERPRETATION LIMIT
A pass shows that a composable relational structure exists inside the token
system and survives a change of language. It does not show that arbitrary
natural text has this structure. The lexicon is hand-built and the operation
set is small. Mixed results stay mixed.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 1b")
        self.geometry("1180x850")
        self.minsize(980, 720)
        self.events = queue.Queue()
        self.worker = None
        self.build()
        self.after(100, self._poll)

    def build(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Experiment 1b — Language-Internal Relational Structure",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Derivational chains in English → frozen relation machinery → German. "
                 "Ground truth is form-to-form; meaning never enters.",
        ).pack(anchor="w", pady=(0, 10))

        cfg = ttk.LabelFrame(outer, text="Locked run", padding=10)
        cfg.pack(fill="x")

        self.seed = tk.StringVar(value="61")
        self.eng_steps = tk.StringVar(value="900")
        self.ger_steps = tk.StringVar(value="300")
        self.batch = tk.StringVar(value="64")

        for i, (lab, var) in enumerate([
            ("Seed", self.seed),
            ("English steps", self.eng_steps),
            ("German steps / model / budget", self.ger_steps),
            ("Batch", self.batch),
        ]):
            ttk.Label(cfg, text=lab).grid(row=0, column=i, padx=5, sticky="w")
            ttk.Entry(cfg, textvariable=var, width=20).grid(row=1, column=i, padx=5, sticky="we")
            cfg.columnconfigure(i, weight=1)

        ttk.Label(
            cfg,
            text="Train depths 1-2, test 1-4.  German budgets: 8, 16, 32, 64, 128 chains.",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(6, 0))

        self.composition = tk.StringVar(value="additive")
        comp = ttk.Frame(cfg)
        comp.grid(row=3, column=0, columnspan=4, sticky="w", padx=5, pady=(8, 0))
        ttk.Label(comp, text="Composition:").pack(side="left")
        ttk.Radiobutton(
            comp, text="additive (relations are angles, composition is addition)",
            variable=self.composition, value="additive",
        ).pack(side="left", padx=(8, 12))
        ttk.Radiobutton(
            comp, text="learned (MLP operator)",
            variable=self.composition, value="learned",
        ).pack(side="left")
        ttk.Label(
            cfg,
            text="Additive: closure, associativity and inversion come from the arithmetic, so depth "
                 "extrapolation needs nothing learned. Learned: the operator must induce them from "
                 "examples that never wrap.",
            foreground="#555555",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=5)

        self.cleanup = tk.BooleanVar(value=False)
        self.num_states = tk.StringVar(value="64")
        self.n_angles = tk.StringVar(value="12")
        opt = ttk.Frame(cfg)
        opt.grid(row=5, column=0, columnspan=4, sticky="w", padx=5, pady=(6, 0))
        ttk.Checkbutton(
            opt, text="Discrete closure codebook (learned mode only)", variable=self.cleanup,
        ).pack(side="left")
        ttk.Label(opt, text="   States:").pack(side="left")
        ttk.Entry(opt, textvariable=self.num_states, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(opt, text="Angles:").pack(side="left")
        ttk.Entry(opt, textvariable=self.n_angles, width=6).pack(side="left", padx=(4, 0))

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=10)
        self.run = ttk.Button(row, text="Run Experiment 1b", command=self._start)
        self.run.pack(side="left")
        self.progress = ttk.Progressbar(row, maximum=100, mode="determinate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)
        log_tab = ttk.Frame(nb, padding=8)
        lock_tab = ttk.Frame(nb, padding=8)
        res_tab = ttk.Frame(nb, padding=8)
        nb.add(log_tab, text="Run log")
        nb.add(lock_tab, text="Locked design")
        nb.add(res_tab, text="Result summary")

        self.log = tk.Text(log_tab, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)
        lock = tk.Text(lock_tab, wrap="word", font=("Consolas", 10))
        lock.pack(fill="both", expand=True)
        lock.insert("1.0", LOCK_TEXT)
        lock.configure(state="disabled")
        self.summary = tk.Text(res_tab, wrap="word", font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)

        self._append("Ready. CPU-only.\n\n")

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            seed = int(self.seed.get())
            es = int(self.eng_steps.get())
            gs = int(self.ger_steps.get())
            batch = int(self.batch.get())
            nstates = int(self.num_states.get())
            nangles = int(self.n_angles.get())
            if es < 50 or gs < 20 or batch < 8 or nstates < 2 or nangles < 2:
                raise ValueError("Use at least 50 English steps, 20 German steps, "
                                 "batch >= 8, states >= 2, angles >= 2.")
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        cleanup = bool(self.cleanup.get())
        composition = self.composition.get()
        self.run.configure(state="disabled")
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        self.summary.delete("1.0", "end")
        self._append("Starting Experiment 1b...\n\n")
        self.worker = threading.Thread(
            target=self._worker, args=(seed, es, gs, batch, cleanup, nstates, composition, nangles), daemon=True
        )
        self.worker.start()

    def _worker(self, seed, es, gs, batch, cleanup, nstates, composition, nangles):
        try:
            base = (f"additive_{nangles}angles" if composition == "additive"
                    else (f"learned_closure{nstates}" if cleanup else "learned_nocleanup"))
            arm = f"{base}_eng{es}_ger{gs}_seed{seed}"
            result = run_experiment(
                output_dir=f"morphology_results/{arm}",
                seed=seed,
                english_steps=es,
                german_steps=gs,
                batch_size=batch,
                cleanup=cleanup,
                num_states=nstates,
                composition=composition,
                n_angles=nangles,
                event_callback=lambda k, p: self.events.put((k, p)),
            )
            self.events.put(("complete", result))
        except Exception as e:
            self.events.put(("error", repr(e)))

    def _poll(self):
        try:
            while True:
                kind, p = self.events.get_nowait()
                if isinstance(p, dict) and "overall" in p:
                    self.progress["value"] = 100 * p["overall"]

                if kind == "english_train":
                    self._append(
                        f"English {p['model']:12s} {p['step']:4d}/{p['total']} | "
                        f"loss={p['loss']:.5f} | acc={p['acc']:.3f}\n"
                    )
                elif kind == "depth_done":
                    r = p["row"]
                    self._append(
                        f"\n  DEPTH {r['depth']}  relational={r['relational']:.3f}  "
                        f"transformer={r['transformer']:.3f}  majority={r['majority']:.3f}  "
                        f"(n={r['n']})\n"
                    )
                elif kind == "budget_done":
                    r = p["row"]
                    self._append(
                        f"\n  GERMAN n={r['examples']:3d}  transfer={r['transfer']:.3f}  "
                        f"scratch={r['scratch']:.3f}  scrambled={r['scrambled']:.3f}  "
                        f"transformer={r['transformer']:.3f}  majority={r['majority']:.3f}\n"
                    )
                elif kind in ("status", "done_status"):
                    self._append("\n" + p["text"] + "\n")
                elif kind == "complete":
                    self.progress["value"] = 100
                    self.run.configure(state="normal")
                    report = p["report"]
                    self._append("\nExperiment complete.\nResults folder:\n" + p["output_dir"] + "\n")
                    compact = {
                        "composition": report["composition"],
                        "n_angles": report["n_angles"],
                        "state_cleanup": report["state_cleanup"],
                        "num_relation_states": report["num_relation_states"],
                        "composed_relation_space": report["composed_relation_space"],
                        "depth_generalization": report["depth_generalization"],
                        "crosslingual_transfer": report["crosslingual_transfer"],
                        "parameter_counts": report["parameter_counts"],
                        "criterion_results": report["criterion_results"],
                        "all_locked_criteria_met": report["all_locked_criteria_met"],
                        "interpretation_rule": report["interpretation_rule"],
                    }
                    self.summary.insert("1.0", json.dumps(compact, indent=2))
                elif kind == "error":
                    self.run.configure(state="normal")
                    self._append("\nERROR:\n" + p + "\n")
                    messagebox.showerror("Experiment failed", p)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
