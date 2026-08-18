from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from experiment import run_experiment
from relational_core import topology_change_demo


class RelationalTokenLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — U(1)/Z4 Prototype")
        self.geometry("1050x760")
        self.minsize(900, 650)

        self.events = queue.Queue()
        self.worker = None

        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="Relational Token Lab",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            outer,
            text=(
                "CPU-only toy test: exact complex closure vs learned phase composition "
                "vs a tiny transformer."
            ),
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(outer, text="Experiment settings", padding=10)
        controls.pack(fill="x")

        self.seed_var = tk.StringVar(value="7")
        self.train_len_var = tk.StringVar(value="5")
        self.test_len_var = tk.StringVar(value="32")
        self.steps_var = tk.StringVar(value="700")
        self.batch_var = tk.StringVar(value="192")
        self.examples_var = tk.StringVar(value="500")

        fields = [
            ("Seed", self.seed_var),
            ("Max training path length", self.train_len_var),
            ("Max test path length", self.test_len_var),
            ("Training steps", self.steps_var),
            ("Batch size", self.batch_var),
            ("Test examples / length", self.examples_var),
        ]

        for i, (label, var) in enumerate(fields):
            ttk.Label(controls, text=label).grid(
                row=i // 3 * 2, column=i % 3, sticky="w", padx=6, pady=(2, 0)
            )
            ttk.Entry(controls, textvariable=var, width=18).grid(
                row=i // 3 * 2 + 1, column=i % 3, sticky="we", padx=6, pady=(0, 6)
            )

        for col in range(3):
            controls.columnconfigure(col, weight=1)

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=10)

        self.run_button = ttk.Button(
            button_row, text="Run full experiment", command=self._start_experiment
        )
        self.run_button.pack(side="left")

        ttk.Button(
            button_row, text="Run topology demo only", command=self._run_topology_only
        ).pack(side="left", padx=(8, 0))

        ttk.Button(
            button_row, text="Open results folder path", command=self._show_results_path
        ).pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(
            button_row, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        log_tab = ttk.Frame(notebook, padding=8)
        summary_tab = ttk.Frame(notebook, padding=8)
        about_tab = ttk.Frame(notebook, padding=8)
        notebook.add(log_tab, text="Run log")
        notebook.add(summary_tab, text="Summary")
        notebook.add(about_tab, text="What this test means")

        self.log = tk.Text(log_tab, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        self.summary = tk.Text(summary_tab, wrap="word", font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)

        about = tk.Text(about_tab, wrap="word", font=("Segoe UI", 10))
        about.pack(fill="both", expand=True)
        about.insert(
            "1.0",
            """WHAT THIS APP TESTS

The toy universe has four directed relations represented by quarter-turn complex phases:

    1, i, -1, -i

A path composes relations by complex multiplication, equivalently addition modulo 4.

Three systems receive the same synthetic path task:

1. Exact relational closure
   The correct algebra is explicitly built in. This is the hypothesis in its pure form.

2. Learned phase composition
   The model must learn the phase associated with each relation symbol, but composition
   is structurally constrained to angle addition.

3. Tiny transformer
   A generic transformer receives the same relation symbols and must learn the entire
   input/output computation from short examples.

Training is limited to short paths. Testing extends to longer paths.

The second test creates loops. A loop is consistent exactly when its net phase closes
to the identity. This turns relational closure into contradiction detection.

The topology demo adds and removes graph relations while the program is running and
recomputes implications without retraining.

WHAT A POSITIVE RESULT WOULD MEAN

It would show that explicitly encoding a compositional relational geometry provides
the expected inductive bias on this synthetic task, especially outside the training
path lengths.

WHAT IT WOULD NOT MEAN

It would NOT prove that complex relational coordinates replace transformer attention,
solve natural language, or constitute AGI. The exact relational engine is given the
correct algebra and therefore has a structural advantage by design. The useful question
is whether that advantage can later be learned from language while retaining reliable
composition and topology change.

SAFETY

This program is deliberately local and CPU-only. It has no networking code, no browser
control, no agent loop, no self-modification, and no ability to execute generated
commands. It only generates synthetic relation sequences, trains two tiny local models,
and writes plots/CSV/report files into the local 'results' folder.
""",
        )
        about.configure(state="disabled")

        self._append_log("Ready. Default run is CPU-only.\n")

    def _values(self):
        seed = int(self.seed_var.get())
        train_max = int(self.train_len_var.get())
        test_max = int(self.test_len_var.get())
        steps = int(self.steps_var.get())
        batch = int(self.batch_var.get())
        examples = int(self.examples_var.get())

        if train_max < 1:
            raise ValueError("Training max length must be >= 1.")
        if test_max <= train_max:
            raise ValueError("Test max length must be greater than training max length.")
        if test_max > 128:
            raise ValueError("Keep max test length <= 128 for this toy app.")
        if steps < 10:
            raise ValueError("Training steps must be >= 10.")
        if batch < 8:
            raise ValueError("Batch size must be >= 8.")
        if examples < 20:
            raise ValueError("Test examples per length must be >= 20.")
        return seed, train_max, test_max, steps, batch, examples

    def _start_experiment(self):
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
        self._append_log("Starting experiment...\n")

        self.worker = threading.Thread(
            target=self._experiment_worker, args=values, daemon=True
        )
        self.worker.start()

    def _experiment_worker(
        self, seed, train_max, test_max, steps, batch, examples
    ):
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

            result = run_experiment(
                output_dir="results",
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
                    self._append_log(
                        f"{payload['step']:4d}/{payload['total']} | "
                        f"phase loss={payload['phase_loss']:.5f} | "
                        f"transformer loss={payload['transformer_loss']:.5f}\n"
                    )

                elif kind == "done":
                    self.progress["value"] = 100
                    self.run_button.configure(state="normal")
                    self._append_log("\nTraining and evaluation complete.\n\n")
                    self._append_log(payload["topology_demo"])
                    self._append_log("\n\nResults folder:\n" + payload["output_dir"])

                    report = payload["report"]
                    self.summary.delete("1.0", "end")
                    self.summary.insert(
                        "1.0",
                        json.dumps(
                            {
                                "generalization_summary": report["generalization_summary"],
                                "contradiction_summary": report["contradiction_summary"],
                                "learned_phase_angles_radians_raw":
                                    report["learned_phase_angles_radians_raw"],
                                "interpretation_warning": report["interpretation_warning"],
                            },
                            indent=2,
                        ),
                    )

                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self._append_log("\nERROR:\n" + payload + "\n")
                    messagebox.showerror("Experiment failed", payload)

        except queue.Empty:
            pass

        self.after(100, self._poll_events)

    def _run_topology_only(self):
        try:
            seed = int(self.seed_var.get())
            text = topology_change_demo(seed)
            self.log.delete("1.0", "end")
            self._append_log(text)
        except Exception as e:
            messagebox.showerror("Topology demo failed", str(e))

    def _show_results_path(self):
        path = str(Path("results").resolve())
        messagebox.showinfo(
            "Results folder",
            path + "\n\nThe app does not launch external programs automatically.",
        )

    def _append_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")


if __name__ == "__main__":
    RelationalTokenLabApp().mainloop()
