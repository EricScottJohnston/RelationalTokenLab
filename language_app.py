from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from language_experiment import run_language_experiment


class LanguageExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 1: Language to Geometry")
        self.geometry("1050x760")
        self.minsize(900, 650)

        self.events = queue.Queue()
        self.worker = None
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Experiment 1 — Language → Relational Geometry",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Train only on short natural-language relation chains, then test "
                "whether learned complex relations compose at unseen depths."
            ),
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(outer, text="Settings", padding=10)
        controls.pack(fill="x")

        self.seed_var = tk.StringVar(value="11")
        self.train_len_var = tk.StringVar(value="5")
        self.test_len_var = tk.StringVar(value="32")
        self.steps_var = tk.StringVar(value="900")
        self.batch_var = tk.StringVar(value="128")
        self.examples_var = tk.StringVar(value="400")

        fields = [
            ("Seed", self.seed_var),
            ("Max training relations", self.train_len_var),
            ("Max test relations", self.test_len_var),
            ("Training steps", self.steps_var),
            ("Batch size", self.batch_var),
            ("Test examples / length", self.examples_var),
        ]

        for i, (label, var) in enumerate(fields):
            row = (i // 3) * 2
            col = i % 3
            ttk.Label(controls, text=label).grid(row=row, column=col, sticky="w", padx=6)
            ttk.Entry(controls, textvariable=var, width=18).grid(
                row=row + 1, column=col, sticky="we", padx=6, pady=(0, 6)
            )

        for c in range(3):
            controls.columnconfigure(c, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=10)

        self.run_button = ttk.Button(
            buttons, text="Run language experiment", command=self._start
        )
        self.run_button.pack(side="left")

        self.progress = ttk.Progressbar(
            buttons, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        log_tab = ttk.Frame(notebook, padding=8)
        summary_tab = ttk.Frame(notebook, padding=8)
        meaning_tab = ttk.Frame(notebook, padding=8)
        notebook.add(log_tab, text="Run log")
        notebook.add(summary_tab, text="Summary")
        notebook.add(meaning_tab, text="What this experiment tests")

        self.log = tk.Text(log_tab, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        self.summary = tk.Text(summary_tab, wrap="word", font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)

        meaning = tk.Text(meaning_tab, wrap="word", font=("Segoe UI", 10))
        meaning.pack(fill="both", expand=True)
        meaning.insert(
            "1.0",
            """The previous experiment gave the model relation IDs directly.

This experiment inserts a learned language layer.

Input phrases include forms such as:

    "a quarter turn clockwise from"
    "turned right ninety degrees from"
    "opposite to"
    "facing the reverse direction from"
    "a quarter turn counterclockwise from"

The relational model is NOT told which numeric phase each phrase means.
It learns a unit complex relation from the words using only final path labels.

It then composes the learned edge relations by complex multiplication.

The baseline transformer receives the same words and final labels, but no explicit
composition law.

Training path length stops at 5. Testing continues to 32.

What we want to know:

1. Can the relational model learn the language -> phase mapping?
2. Once learned, does structural composition remain accurate beyond training depth?
3. Does loop closure still provide contradiction detection when the inputs are words
   rather than preassigned relation IDs?

This is still a controlled synthetic language experiment, not unrestricted NLP.
""",
        )
        meaning.configure(state="disabled")

        self._append("Ready. Leave the defaults for the first run.\n")

    def _values(self):
        seed = int(self.seed_var.get())
        train_max = int(self.train_len_var.get())
        test_max = int(self.test_len_var.get())
        steps = int(self.steps_var.get())
        batch = int(self.batch_var.get())
        examples = int(self.examples_var.get())

        if train_max < 1:
            raise ValueError("Training max must be >= 1.")
        if test_max <= train_max:
            raise ValueError("Test max must be greater than training max.")
        if test_max > 64:
            raise ValueError("Keep test max <= 64 for this CPU experiment.")
        if steps < 50:
            raise ValueError("Use at least 50 training steps.")
        if batch < 8:
            raise ValueError("Batch size must be >= 8.")
        if examples < 20:
            raise ValueError("Examples per length must be >= 20.")
        return seed, train_max, test_max, steps, batch, examples

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            values = self._values()
        except Exception as e:
            messagebox.showerror("Invalid setting", str(e))
            return

        self.run_button.configure(state="disabled")
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        self.summary.delete("1.0", "end")
        self._append("Starting Experiment 1...\n")

        self.worker = threading.Thread(
            target=self._worker, args=values, daemon=True
        )
        self.worker.start()

    def _worker(self, seed, train_max, test_max, steps, batch, examples):
        try:
            def progress(step, total, phase_loss, transformer_loss):
                self.events.put(
                    (
                        "progress",
                        {
                            "step": step,
                            "total": total,
                            "phase_loss": phase_loss,
                            "transformer_loss": transformer_loss,
                        },
                    )
                )

            result = run_language_experiment(
                output_dir="language_results",
                seed=seed,
                train_max_len=train_max,
                test_max_len=test_max,
                steps=steps,
                batch_size=batch,
                examples_per_length=examples,
                progress_callback=progress,
            )
            self.events.put(("done", result))
        except Exception as e:
            self.events.put(("error", repr(e)))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()

                if kind == "progress":
                    pct = payload["step"] / payload["total"] * 100
                    self.progress["value"] = pct
                    self._append(
                        f"{payload['step']:4d}/{payload['total']} | "
                        f"language-phase loss={payload['phase_loss']:.5f} | "
                        f"transformer loss={payload['transformer_loss']:.5f}\n"
                    )

                elif kind == "done":
                    self.progress["value"] = 100
                    self.run_button.configure(state="normal")
                    self._append("\nExperiment complete.\n")
                    self._append("Results folder:\n" + payload["output_dir"] + "\n")

                    report = payload["report"]
                    summary = {
                        "generalization_summary": report["generalization_summary"],
                        "contradiction_summary": report["contradiction_summary"],
                        "phrase_phase_table": report["phrase_phase_table"],
                        "interpretation_warning": report["interpretation_warning"],
                    }
                    self.summary.insert("1.0", json.dumps(summary, indent=2))

                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self._append("\nERROR:\n" + payload + "\n")
                    messagebox.showerror("Experiment failed", payload)

        except queue.Empty:
            pass

        self.after(100, self._poll_events)

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")


if __name__ == "__main__":
    LanguageExperimentApp().mainloop()
