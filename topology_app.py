from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from topology_experiment import run_topology_experiment


class TopologyExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relational Token Lab — Experiment 2: Changing Topology")
        self.geometry("1080x780")
        self.minsize(920, 680)

        self.events = queue.Queue()
        self.worker = None

        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Experiment 2 — Changing Relational Topology",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Train learned baselines only on static 4–8 node graphs; then change "
                "relational boundaries on graphs up to 24 nodes without retraining."
            ),
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(outer, text="Settings", padding=10)
        controls.pack(fill="x")

        self.seed_var = tk.StringVar(value="19")
        self.steps_var = tk.StringVar(value="850")
        self.batch_var = tk.StringVar(value="96")
        self.episodes_var = tk.StringVar(value="250")

        fields = [
            ("Seed", self.seed_var),
            ("Training steps", self.steps_var),
            ("Batch size", self.batch_var),
            ("Episodes per graph size", self.episodes_var),
        ]

        for i, (label, var) in enumerate(fields):
            ttk.Label(controls, text=label).grid(
                row=0, column=i, sticky="w", padx=6
            )
            ttk.Entry(controls, textvariable=var, width=18).grid(
                row=1, column=i, sticky="we", padx=6, pady=(0, 6)
            )
            controls.columnconfigure(i, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=10)

        self.run_button = ttk.Button(
            buttons,
            text="Run topology experiment",
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
        notebook.add(explain_tab, text="What this test does")

        self.log = tk.Text(log_tab, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        self.summary = tk.Text(summary_tab, wrap="word", font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)

        explain = tk.Text(explain_tab, wrap="word", font=("Segoe UI", 10))
        explain.pack(fill="both", expand=True)
        explain.insert(
            "1.0",
            """EXPERIMENT 2 ISOLATES THE TOPOLOGY HYPOTHESIS

Experiment 1 already showed that the language encoder could map controlled phrases
into the four relational phases. This test removes language on purpose so we are
testing only the changing-relational-boundary idea.

All systems receive the same edge relations 0, +90, 180, -90.

The exact relational engine is the structural hypothesis.

Two learned baselines are included:

1. Transformer baseline
   Receives the graph as a sequence of edge records and a source/target query.

2. GNN baseline
   Receives the graph directly and performs four rounds of ordinary learned
   message passing.

The learned models train only on STATIC graphs with 4–8 nodes.

Then, with NO RETRAINING, each test episode uses a larger graph and changes it:

BASE
    Source and target are connected through a coherent relational tree.

CUT
    One relationship is removed from their path. They become disconnected.
    Correct answer becomes UNKNOWN.

RECONNECT
    A new, correct relational edge reconnects the components.
    The old source→target relation becomes derivable again.

CONTRADICTION
    A wrong edge is added. The graph now contains a coherence defect.

REPAIR
    The wrong edge is removed. Coherence returns.

The same inference code/models are used at every state.

What would be interesting:

- exact relational engine changes answers immediately when topology changes;
- exact relation/coherence accuracy stays at 100% regardless of graph size;
- learned baselines degrade on larger topologies or after relation changes.

This is still a synthetic Z4 relational world. A positive result would support the
architecture's structural treatment of changing boundaries, not prove general AI.
""",
        )
        explain.configure(state="disabled")

        self._append("Ready. Leave defaults alone for the first run.\n")

    def _values(self):
        seed = int(self.seed_var.get())
        steps = int(self.steps_var.get())
        batch = int(self.batch_var.get())
        episodes = int(self.episodes_var.get())

        if steps < 100:
            raise ValueError("Use at least 100 training steps.")
        if batch < 16:
            raise ValueError("Batch size must be >= 16.")
        if episodes < 25:
            raise ValueError("Episodes per size must be >= 25.")

        return seed, steps, batch, episodes

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
        self._append("Starting Experiment 2...\n")

        self.worker = threading.Thread(
            target=self._worker,
            args=values,
            daemon=True,
        )
        self.worker.start()

    def _worker(self, seed, steps, batch, episodes):
        try:
            def progress(step, total, transformer_loss, gnn_loss):
                self.events.put(
                    (
                        "progress",
                        {
                            "step": step,
                            "total": total,
                            "transformer_loss": transformer_loss,
                            "gnn_loss": gnn_loss,
                        },
                    )
                )

            result = run_topology_experiment(
                output_dir="topology_results",
                seed=seed,
                steps=steps,
                batch_size=batch,
                episodes_per_size=episodes,
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
                        f"transformer loss={payload['transformer_loss']:.5f} | "
                        f"GNN loss={payload['gnn_loss']:.5f}\n"
                    )

                elif kind == "done":
                    self.progress["value"] = 100
                    self.run_button.configure(state="normal")
                    self._append("\nExperiment complete.\n\n")
                    self._append(payload["episode_text"])
                    self._append("\nResults folder:\n" + payload["output_dir"] + "\n")

                    self.summary.insert(
                        "1.0",
                        json.dumps(
                            {
                                "size_summary": payload["report"]["size_summary"],
                                "dynamic_sequence": payload["report"]["dynamic_sequence"],
                                "interpretation": payload["report"]["interpretation"],
                            },
                            indent=2,
                        ),
                    )

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
    TopologyExperimentApp().mainloop()
