Relational Token Lab

A deliberately small, CPU-only experiment for testing a complex relational
composition idea before spending money on GPU compute.

What it tests

The synthetic relation set is:

1

i

-1

-i

These are the four quarter-turn elements of a simple U(1) / Z4 relation
system. Relations compose by complex multiplication.

For example:

A --i--> B
B --i--> C

therefore

A ---(-1)---> C

because:

i * i = -1

The experiment compares:

Exact relational closure — the relation algebra is explicitly known.

Learned phase composition — the model learns the phase associated with
each relation symbol, but path composition is structurally constrained.

Tiny transformer baseline — the model has no explicit relation algebra
and must learn the task from examples.

The models are trained only on short paths and then evaluated on paths much
longer than anything seen during training.

A second evaluation treats a relational loop as a contradiction test:
a loop is consistent only if its net phase returns to the identity.

A third non-neural demo changes a relational graph while it is running by
adding/removing edges and checks whether implications update without retraining.

Safety / scope

This is not an autonomous agent.

The project contains:

no networking code,

no browser automation,

no shell/subprocess execution,

no model downloads,

no code generation,

no self-modification,

no mechanism for executing actions produced by the models.

It generates synthetic integer relation sequences, trains two tiny local
models, and writes experiment results to a local results folder.

Visual Studio setup

Install Python 3.11 or newer if needed.

In Visual Studio, make sure the Python development workload is installed.

Open this folder in Visual Studio.

Open a terminal in the project folder.

Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

python -m pip install -r requirements.txt

Run the desktop app:

python app.py

You can also double-click run_windows.bat after the environment/dependencies
are configured, provided python resolves to the intended interpreter.

First run

Use the defaults:

training max path length: 5

test max path length: 32

training steps: 700

batch size: 192

test examples per length: 500

Click Run full experiment.

The app writes:

results/
    generalization.csv
    contradictions.csv
    generalization.png
    contradictions.png
    topology_demo.txt
    report.json
    models.pt

The two PNG files are the easiest first look.

Command-line run

If you prefer no GUI:

python experiment.py

Unit tests

python -m unittest test_core.py

How to interpret the result

The important region is to the right of the dashed line in the plots.
That is path length beyond the training distribution.

A strong result would look like:

exact relational closure remains at or near 100%;

learned phase composition remains high after training only on short paths;

the generic transformer degrades as reasoning depth extends beyond training.

That would demonstrate an inductive-bias advantage on this synthetic task.

It would not establish that the architecture replaces transformer attention
for natural language. The exact engine is handed the correct composition law,
which is an intentional structural advantage. The next scientific question
would be whether useful relations and their composition can be learned from
natural-language data while preserving the same closure properties.

Why the topology demo matters

The graph demo separates two ideas:

state changes inside a fixed relational graph

from:

the relational graph itself changes

Adding or removing a relation changes what can be inferred without changing
the inference algorithm. That is the minimal prototype of the
"changing relational boundary" idea.
