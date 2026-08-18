from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from operator_experiment import run_operator_experiment


class OperatorExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 3: Learn the Composition Law")
        self.geometry("1120x800")
        self.minsize(940, 700)

        self.events = queue.Queue()
        self.worker = None

        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Experiment 3 — Learn the Composition Law",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "The relational model no longer receives complex multiplication. "
                "A neural binary operator must learn how relations compose."
            ),
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(outer, text="Settings", padding=10)
        controls.pack(fill="x")

        self.seed_var = tk.StringVar(value="29")
        self.steps_var = tk.StringVar(value="950")
        self.batch_var = tk.StringVar(value="160")
        self.examples_var = tk.StringVar(value="400")
        self.topology_var = tk.StringVar(value="180")

        fields = [
            ("Seed", self.seed_var),
            ("Training steps", self.steps_var),
            ("Batch size", self.batch_var),
            ("Examples / depth", self.examples_var),
            ("Topology episodes / size", self.topology_var),
        ]

        for i, (label, var) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=i, sticky="w", padx=5)
            ttk.Entry(controls, textvariable=var, width=16).grid(
                row=1, column=i, sticky="we", padx=5, pady=(0, 6)
            )
            controls.columnconfigure(i, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=10)

        self.run_button = ttk.Button(
            buttons,
            text="Run learned-operator experiment",
            command=self._start,
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
        explain_tab = ttk.Frame(notebook, padding=8)

        notebook.add(log_tab, text="Run log")
        notebook.add(summary_tab, text="Summary")
        notebook.add(explain_tab, text="What changed")

        self.log = tk.Text(log_tab, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        self.summary = tk.Text(summary_tab, wrap="word", font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)

        explain = tk.Text(explain_tab, wrap="word", font=("Segoe UI", 10))
        explain.pack(fill="both", expand=True)
        explain.insert(
            "1.0",
            """THIS IS THE IMPORTANT CHANGE

Experiments 0–2 supplied the relational engine with the composition law.

Experiment 3 does not.

The hidden synthetic world still has four relations. The DATA GENERATOR knows the
correct answers, but the learned relational models do not receive the Z4 table,
complex multiplication, or modular addition.

Instead, each learned relational model has:

    four learned relation vectors
    +
    a neural binary operator F(a,b)
    +
    recursive composition

Two versions are compared:

1. STRUCTURED LEARNED OPERATOR

   It gets algebraic requirements, not the answer table:

       identity
       inverse consistency
       associativity
       closure onto the relation-state set

   It must discover a composition table that both fits shallow examples and satisfies
   those structural constraints.

2. UNCONSTRAINED LEARNED OPERATOR

   Same embeddings and same neural binary operator, but only task supervision.

3. TINY TRANSFORMER

   Receives the same short relation sequences and final labels.

TRAINING

    composition depths 1–5 only

TEST

    composition depths 1–64

Then the learned operators are placed into changing-topology graph episodes, without
additional training, to see whether the learned law continues to work when relational
boundaries are cut, reconnected, contradicted, and repaired.

The report will print the composition table the model actually discovered.

WHAT WOULD BE VERY INTERESTING

If the structured model:

    - learns the hidden 4x4 composition table correctly,
    - remains highly accurate far beyond depth 5,
    - stays coherent on changing topologies,
    - while the unconstrained operator or transformer degrades,

then the reusable relational law was not directly supplied. It emerged from shallow
examples plus structural coherence requirements.

SCIENTIFIC LIMIT

This is still a synthetic four-relation world. We are not asking the system to invent
arbitrary mathematics from unrestricted data.
""",
        )
        explain.configure(state="disabled")

        self._append("Ready. Default run is CPU-only. Progress now includes evaluation.\n")

    def _values(self):
        seed = int(self.seed_var.get())
        steps = int(self.steps_var.get())
        batch = int(self.batch_var.get())
        examples = int(self.examples_var.get())
        topology = int(self.topology_var.get())

        if steps < 100:
            raise ValueError("Use at least 100 training steps.")
        if batch < 16:
            raise ValueError("Batch size must be >= 16.")
        if examples < 50:
            raise ValueError("Examples per depth must be >= 50.")
        if topology < 25:
            raise ValueError("Topology episodes per size must be >= 25.")

        return seed, steps, batch, examples, topology

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
        self._append("Starting Experiment 3...\n")

        self.worker = threading.Thread(
            target=self._worker, args=values, daemon=True
        )
        self.worker.start()

    def _worker(self, seed, steps, batch, examples, topology):
        try:
            def event_callback(kind, payload):
                self.events.put((kind, payload))

            result = run_operator_experiment(
                output_dir="operator_results",
                seed=seed,
                steps=steps,
                batch_size=batch,
                examples_per_length=examples,
                topology_episodes_per_size=topology,
                event_callback=event_callback,
            )
            self.events.put(("done", result))

        except Exception as e:
            self.events.put(("error", repr(e)))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()

                if kind == "train":
                    self.progress["value"] = payload["overall_pct"] * 100
                    self._append(
                        f"{payload['step']:4d}/{payload['total']} | "
                        f"structured task={payload['structured_task_loss']:.5f} | "
                        f"structured total={payload['structured_total_loss']:.5f} | "
                        f"unconstrained={payload['unconstrained_loss']:.5f} | "
                        f"transformer={payload['transformer_loss']:.5f}\n"
                    )
                    if payload["step"] % 100 == 0:
                        r = payload["regs"]
                        self._append(
                            "    structural residuals: "
                            f"id={r['identity']:.5f}, inv={r['inverse']:.5f}, "
                            f"assoc={r['associativity']:.5f}, closure={r['closure']:.5f}\n"
                        )

                elif kind == "depth":
                    self.progress["value"] = payload["overall_pct"] * 100
                    self._append(
                        f"Depth evaluation: {payload['done']}/{payload['total']}\n"
                    )

                elif kind == "topology":
                    self.progress["value"] = payload["overall_pct"] * 100
                    self._append(
                        f"Topology evaluation: {payload['done']}/{payload['total']} "
                        f"(currently {payload['node_count']} nodes)\n"
                    )

                elif kind == "status":
                    self._append("\n" + payload["text"] + "\n")

                elif kind == "done":
                    self.progress["value"] = 100
                    self.run_button.configure(state="normal")
                    self._append("\nExperiment complete.\n")
                    self._append("Results folder:\n" + payload["output_dir"] + "\n\n")
                    self._append(payload["table_text"])

                    report = payload["report"]
                    compact = {
                        "critical_design_fact": report["critical_design_fact"],
                        "depth_summary": report["depth_summary"],
                        "structured_operator": {
                            "predicted_cayley_table":
                                report["structured_operator"]["predicted_cayley_table"],
                            "table_accuracy":
                                report["structured_operator"]["table_accuracy"],
                            "class_associativity_accuracy":
                                report["structured_operator"]["class_associativity_accuracy"],
                            "structural_residuals":
                                report["structured_operator"]["structural_residuals"],
                        },
                        "unconstrained_operator": {
                            "predicted_cayley_table":
                                report["unconstrained_operator"]["predicted_cayley_table"],
                            "table_accuracy":
                                report["unconstrained_operator"]["table_accuracy"],
                        },
                        "topology_summary": report["topology_summary"],
                        "interpretation_warning": report["interpretation_warning"],
                    }
                    self.summary.insert("1.0", json.dumps(compact, indent=2))

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
    OperatorExperimentApp().mainloop()
