from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from bootstrap_experiment import run_experiment

LOCK_TEXT = """EXPERIMENT 7 — TEACHING THE MACHINE WORDS IT DOES NOT KNOW
Locked before the run.

WHERE THIS CAME FROM
Experiment 6 said two things clearly.

  It reads a relation fine when it recognises the word.
  It cannot read one when there is no word to recognise.

and one more, which is the reason this experiment exists:

  Give it two sentences it CAN read, joined at a shared thing, and ask it to
  add them up — it got that right 998 times out of 1000, having never been
  trained on pairs.

The adding is exact. So use the adding backwards.

THE IDEA IN ONE PICTURE

  kleen  --raises-->  spunt  --gorbles-->  frell      <- gorbles is unknown
  kleen  --reduces-->                      frell      <- this you can read

  up, then gorbles, gets you to down.
  So gorbles must be down.

Nobody had to define "gorbles". The rest of the text defined it.

SCALED UP, IT IS ALGEBRA
Every sentence the machine understands is one equation. Every sentence with an
unknown word in it is an equation with one extra unknown. Hundreds of
sentences give hundreds of equations, and you solve the whole system at once.
Because relations only ever go two ways — promoting or suppressing — the
arithmetic is the simplest there is: everything is a one or a zero and adding
means XOR.

The solving is ordinary Gaussian elimination, the thing you would do to any
system of equations, run over ones and zeros instead of decimals.

WHAT ABOUT THE TYPE, NOT JUST THE DIRECTION
Direction (up or down) is a group and adds. Type (cause, part-of, time order,
size) does not add — but it is a constraint: you can only chain relations of
the same type. So the machine tries the unknown word in all four type-systems
and keeps the one it does not break. A word that fits in more than one is left
alone.

This matters, because type is exactly what Experiment 6's reader was worst at.
The reader saw type collapse to near chance on unfamiliar wording. This
mechanism gets type from consistency across many sentences instead of from one
sentence, so it may recover the very thing the reader could not.

IT IS ALLOWED TO SAY "I DON'T KNOW"
After the elimination, some unknowns are pinned down and some are not. A word
that the text does not determine comes back unclaimed. Not a guess with a low
score — unclaimed. This is the same behaviour the lexicon already has when two
relations cross types: there is no product, so there is no answer.

WHAT IS BEING MEASURED

  A  Hide 10%, 30%, 50%, 70%, 90% of the lexicon and see how much comes back.

     The number to watch is where it breaks. Below some amount of known
     vocabulary there is not enough left to triangulate against and the whole
     thing goes underdetermined. Where that happens is the practical answer to
     "how big does the starter lexicon have to be before it can grow itself?"

  B  Does the loop pay for itself. Every word identified becomes a known word,
     which makes new equations, which may identify more. Round 2 and beyond
     either finds things or it does not.

  C  What happens when the reader is wrong. A misread sentence contradicts the
     others. Does the system survive it, and does it notice?

CONTROLS
  random       give each unknown word a random type and direction
  majority     give them all the commonest answer
  scrambled    THE IMPORTANT ONE. Build the text so the relations do NOT add
               up consistently — same words, same sentences, incoherent world.
               The equations now have no solution and recovery must collapse.
               If it does not collapse, then something other than the algebra
               is producing the result, and the finding is worthless.

LOCKED SUCCESS CRITERIA
  at 30% hidden, type and direction both right          >= 0.80
  at 30% hidden, right when it commits                  >= 0.95
  at 50% hidden, recovered                              >= 0.60
  at 30% hidden, beats random by                        >= 0.50
  at 30% hidden, beats majority class by                >= 0.40
  scrambled text at 30% hidden recovers                 <= 0.10
  at 50% hidden, round 2 or later finds something       > 0
  at 5% reader error, contradictions found              >= 0.80 of injected
  claims less at 90% hidden than at 10% hidden          true

WHAT A PASS MEANS
The machine can grow its own vocabulary out of text it can only partly read,
using composition as the rule that identifies the missing words, and it
declines to guess when the text does not settle the matter.

WHAT IT DOES NOT MEAN
Only relations that chain can be recovered — a relation that does not compose
carries no equation, and no amount of text will identify it. The text here is
generated, so it is consistent by construction; real prose is not. And a word
that appears in only one or two sentences will not be pinned down no matter
how good the solver is.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 7")
        self.geometry("1180x880")
        self.minsize(1000, 740)
        self.events = queue.Queue()
        self.worker = None
        self.build()
        self.after(100, self._poll)

    def build(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Experiment 7 — Teaching the Machine Words It Does Not Know",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Hide part of the lexicon, then recover it by solving the composition "
                 "backwards. The scrambled control is the one that decides whether the "
                 "algebra is really doing the work.",
        ).pack(anchor="w", pady=(0, 10))

        cfg = ttk.LabelFrame(outer, text="Locked run", padding=10)
        cfg.pack(fill="x")

        self.seed = tk.StringVar(value="91")
        self.entities = tk.StringVar(value="60")
        self.edges = tk.StringVar(value="900")

        for i, (lab, var) in enumerate([("Seed", self.seed),
                                        ("Entities", self.entities),
                                        ("Sentences per type", self.edges)]):
            ttk.Label(cfg, text=lab).grid(row=0, column=i, padx=5, sticky="w")
            ttk.Entry(cfg, textvariable=var, width=18).grid(row=1, column=i, padx=5, sticky="we")
            cfg.columnconfigure(i, weight=1)

        ttk.Label(
            cfg,
            text="Hidden-lexicon sweep: 10%, 30%, 50%, 70%, 90%.  Reader-error runs: 2%, 5%.  "
                 "Fewer sentences per type means fewer equations, so recovery should fall.",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 0))

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=10)
        self.run = ttk.Button(row, text="Run Experiment 7", command=self._start)
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

        self._append("Ready. No neural network in this one — it is a solver.\n\n")

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            seed = int(self.seed.get())
            ents = int(self.entities.get())
            edges = int(self.edges.get())
            if ents < 10 or edges < 20:
                raise ValueError("Use at least 10 entities and 20 sentences per type.")
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.run.configure(state="disabled")
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        self.summary.delete("1.0", "end")
        self._append("Starting Experiment 7...\n\n")
        self.worker = threading.Thread(target=self._worker, args=(seed, ents, edges),
                                       daemon=True)
        self.worker.start()

    def _worker(self, seed, ents, edges):
        try:
            arm = f"e{ents}_s{edges}_seed{seed}"
            result = run_experiment(
                output_dir=f"bootstrap_results/{arm}",
                seed=seed, n_entities=ents, edges_per_kind=edges,
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

                if kind == "sweep_done":
                    r = p["row"]
                    self._append(
                        f"\n  {int(r['hidden_fraction']*100):2d}% OF THE LEXICON HIDDEN "
                        f"({r['hidden_present']} words)\n"
                        f"      recovered            {r['overall']:.3f}\n"
                        f"      claimed at all       {r['identified_fraction']:.3f}\n"
                        f"      right when claimed   {r['accuracy_among_identified']:.3f}\n"
                        f"      left unclaimed       {r['unidentified']}\n"
                        f"      random / majority    {r['random']:.3f} / {r['majority']:.3f}\n"
                        f"      rounds               {r['rounds_used']}  {r['per_round']}\n")
                elif kind == "scrambled_done":
                    r = p["row"]
                    self._append(
                        f"\n  SCRAMBLED CONTROL (incoherent world, 30% hidden)\n"
                        f"      recovered            {r['overall']:.3f}   <- must be near zero\n"
                        f"      claimed at all       {r['identified_fraction']:.3f}\n")
                elif kind == "contradiction_done":
                    r = p["row"]
                    self._append(
                        f"\n  READER ERROR {r['reader_error']:.0%}\n"
                        f"      recovered            {r['overall']:.3f}\n"
                        f"      contradictions found {r['flagged']} of {r['injected']} injected"
                        f"  (ratio {r['detection_ratio']:.2f})\n"
                        f"      of those flagged, actually wrong: "
                        f"{r['attribution_precision']:.2f}\n")
                elif kind in ("status", "done_status"):
                    self._append("\n" + p["text"] + "\n")
                elif kind == "complete":
                    self.progress["value"] = 100
                    self.run.configure(state="normal")
                    report = p["report"]
                    self._append("\nExperiment complete.\nResults folder:\n"
                                 + p["output_dir"] + "\n"
                                 "\nOpen worked_examples.txt in that folder to see the actual "
                                 "words it figured out and the sentences it used.\n")
                    compact = {
                        "hyperparameters": report["hyperparameters"],
                        "sweep": report["sweep"],
                        "scrambled_control": report["scrambled_control"],
                        "contradiction": report["contradiction"],
                        "criterion_results": report["criterion_results"],
                        "all_locked_criteria_met": report["all_locked_criteria_met"],
                        "interpretation_rule": report["interpretation_rule"],
                        "known_limits": report["known_limits"],
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
