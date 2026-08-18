from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from sentence_experiment import run_experiment

LOCK_TEXT = """EXPERIMENT 6 — READING A RELATION OUT OF A SENTENCE
Locked before the run.

WHAT THE MACHINE IS ASKED TO DO
You hand it a sentence with two things named in it. It says how those two
things are related. Two answers come out:

  what type    cause, part-of, time order, size, evidence
  which way    does the first push the second up, or down

Both entity names are nonsense words — "kleen", "spunt" — so the machine can
never win by recognising a topic. Only structure is left.

WHY THIS IS THE SECOND DESIGN
The first version trained on exactly one sentence shape:

  "kleen raises spunt."

and then tested on four shapes it had never seen once: passives, sentences
where no relation word appears at all, and relations stated as nouns. Three of
those four scored below the majority-class floor. That is what an impossible
task looks like. You cannot derive the English passive from first principles;
nobody does, children included — they hear it.

The failure was in the split, not in the machine.

WHAT CHANGED
Every sentence shape now appears in training. What is held back is the
particular wording.

  shape          in training                 held back for testing
  ------------   -------------------------   ---------------------------
  plain          most frames, most words     other frames, other words
  "will raise"   two-thirds of relations     the other third
  passive        two-thirds of relations     the other third
  construction   all but one per relation    the one left out
  noun form      all but one per relation    the one left out

So the question is now answerable: given a sentence pattern the machine has
seen used for other relations, can it read a relation it has not seen said
that way?

THE SEVEN TESTS

  A  familiar        Trained words, trained frame, brand new entities.
                     Sanity check. If this is low, nothing else means
                     anything.

  B  new frame       A trained relation word dropped into a sentence frame
                     held out of training. The cheapest kind of
                     generalization there is.

  C  inflection      "will prevent", "is preventing" — a trained word in a
                     form it was never shown, for a relation whose modal and
                     progressive sentences were held out. The stem is shared,
                     so the characters carry the signal.

  D  synonym         Trained on "prevents", tested on "blocks".

                     THIS IS A CONTROL AND IT IS SUPPOSED TO FAIL. Which
                     sounds mean which things is arbitrary — Saussure's point.
                     There is nothing in the letters of "blocks" that says
                     CAUSAL and NEGATIVE. D is here to mark how high
                     arbitrariness alone can push a score, so that a failure
                     somewhere else can be told apart from a failure here.
                     If D scores well, something is leaking and the whole
                     split needs re-auditing.

  E  construction    A sentence pattern held out of training, for a relation
                     whose other patterns were trained, and with no relation
                     word from the lexicon appearing anywhere:

                       "spunt climbs with kleen."

                     THIS IS THE EXPERIMENT. Passing means the relation was
                     read out of the shape of the sentence, not looked up
                     from a word.

  F  passive         "spunt is raised by kleen." The two entities appear in
                     the reverse order and the relation does not reverse with
                     them. Tested on relations whose passives were held out,
                     so the question is whether the convention transfers.

  G  noun form       "The driver of spunt is kleen." A noun-form template held
                     out, for a relation whose other noun forms were trained.

CONTROLS
  majority     always answer with the most common training class
  transformer  a larger model on identical data, with no relation states
  shuffled     test E with the words in random order. Vocabulary kept, syntax
               destroyed. If the score survives this, the machine is counting
               words and the structural claim is dead.
  span-swap    test A with the two entity markers exchanged. It is now being
               asked about the reverse pair, so the score must fall. If it
               does not, the machine is ignoring the markers and reading no
               direction at all.

COMPOSITION — THE PART THAT TESTS THE TOPOLOGY
Two sentences joined at a shared middle entity:

  kleen raises spunt.        spunt reduces frell.

Read separately. Never trained as a pair. The two answers are added — polarity
is a Z2 group, so adding is XOR — and the result is checked against what the
chain actually implies. This is the same arithmetic that carried Experiment 1b
from 0.000 to 1.000 past its training depth. Here it runs on sentences.

The chain is also run on tier E, where neither sentence contains a relation
word.

LOCKED SUCCESS CRITERIA
  tier A                                     >= 0.85
  tier B                                     >= 0.80
  tier C minus its floor                     >= 0.30
  tier E minus its floor                     >= 0.25
  tier E minus transformer                   >= 0.10
  tier F minus its floor                     >= 0.25
  tier G minus its floor                     >= 0.20
  tier E minus word-shuffled tier E          >= 0.20
  tier A minus span-swapped tier A           >= 0.30
  composed polarity on unseen constructions  >= 0.75

Tier D is not a criterion. It is the arbitrariness marker.

INTERPRETATION LIMIT
Passing E and the composition check means a relation can be read out of
sentence structure and then composed arithmetically. It does not mean this
holds for arbitrary prose: the lexicon is hand-built, the entities are nonce,
and the sentence patterns are enumerated. Mixed results stay mixed.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 6")
        self.geometry("1180x880")
        self.minsize(1000, 740)
        self.events = queue.Queue()
        self.worker = None
        self.build()
        self.after(100, self._poll)

    def build(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Experiment 6 — Reading a Relation out of a Sentence",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Every sentence shape is in training. The wording is what is held out. "
                 "Tier E — a sentence pattern it has never seen, with no relation word in it "
                 "— is the experiment.",
        ).pack(anchor="w", pady=(0, 10))

        cfg = ttk.LabelFrame(outer, text="Locked run", padding=10)
        cfg.pack(fill="x")

        self.seed = tk.StringVar(value="71")
        self.steps = tk.StringVar(value="1200")
        self.batch = tk.StringVar(value="64")
        self.train_size = tk.StringVar(value="9000")
        self.test_size = tk.StringVar(value="600")
        self.n_angles = tk.StringVar(value="12")

        fields = [("Seed", self.seed), ("Training steps", self.steps),
                  ("Batch", self.batch), ("Training sentences", self.train_size),
                  ("Test per tier", self.test_size), ("Angles", self.n_angles)]
        for i, (lab, var) in enumerate(fields):
            ttk.Label(cfg, text=lab).grid(row=0, column=i, padx=5, sticky="w")
            ttk.Entry(cfg, textvariable=var, width=16).grid(row=1, column=i, padx=5, sticky="we")
            cfg.columnconfigure(i, weight=1)

        ttk.Label(
            cfg,
            text="Relations are angles and composition is addition — the same arithmetic that "
                 "carried Experiment 1b past its training depth. Angles sets how many.",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=6, sticky="w", padx=5, pady=(8, 0))

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=10)
        self.run = ttk.Button(row, text="Run Experiment 6", command=self._start)
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
            steps = int(self.steps.get())
            batch = int(self.batch.get())
            train_size = int(self.train_size.get())
            test_size = int(self.test_size.get())
            n_angles = int(self.n_angles.get())
            if steps < 50 or batch < 8 or train_size < 500 or test_size < 50 or n_angles < 2:
                raise ValueError("Use at least 50 steps, batch >= 8, 500 training "
                                 "sentences, 50 test per tier, angles >= 2.")
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.run.configure(state="disabled")
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        self.summary.delete("1.0", "end")
        self._append("Starting Experiment 6...\n\n")
        self.worker = threading.Thread(
            target=self._worker,
            args=(seed, steps, batch, train_size, test_size, n_angles),
            daemon=True)
        self.worker.start()

    def _worker(self, seed, steps, batch, train_size, test_size, n_angles):
        try:
            arm = f"design2_{n_angles}angles_s{steps}_n{train_size}_seed{seed}"
            result = run_experiment(
                output_dir=f"sentence_results/{arm}",
                seed=seed,
                steps=steps,
                batch_size=batch,
                n_angles=n_angles,
                train_size=train_size,
                test_size=test_size,
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

                if kind == "train":
                    self._append(f"{p['model']:12s} {p['step']:5d}/{p['total']} | "
                                 f"loss={p['loss']:.5f} | acc={p['acc']:.3f}\n")
                elif kind == "tier_done":
                    r = p["row"]
                    self._append(
                        f"\n  {r['tier']:16s} resolver={r['resolver']:.3f}  "
                        f"transformer={r['transformer']:.3f}  floor={r['majority']:.3f}  "
                        f"(kind={r['resolver_kind']:.3f} pol={r['resolver_polarity']:.3f}, "
                        f"n={r['n']})\n      {r['description']}\n")
                elif kind == "controls_done":
                    c = p["controls"]
                    self._append(
                        "\n  CONTROLS\n"
                        f"    word-shuffled E : resolver={c['resolver_shuffled_E']:.3f}  "
                        f"transformer={c['transformer_shuffled_E']:.3f}\n"
                        f"    span-swapped A  : resolver={c['resolver_spanswap_A']:.3f}  "
                        f"transformer={c['transformer_spanswap_A']:.3f}\n")
                elif kind == "chain_done":
                    r = p["row"]
                    self._append(
                        f"\n  CHAIN on {r['tier']:16s} composed polarity="
                        f"{r['resolver_polarity']:.3f}  kind={r['resolver_kind']:.3f}  "
                        f"both={r['resolver_both']:.3f}  "
                        f"(transformer polarity={r['transformer_polarity']:.3f}, n={r['n']})\n")
                elif kind in ("status", "done_status"):
                    self._append("\n" + p["text"] + "\n")
                elif kind == "complete":
                    self.progress["value"] = 100
                    self.run.configure(state="normal")
                    report = p["report"]
                    self._append("\nExperiment complete.\nResults folder:\n"
                                 + p["output_dir"] + "\n")
                    compact = {
                        "hyperparameters": report["hyperparameters"],
                        "tiers": report["tiers"],
                        "controls": report["controls"],
                        "composition": report["composition"],
                        "parameter_counts": report["parameter_counts"],
                        "arbitrariness_control": report["arbitrariness_control"],
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
