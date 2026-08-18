# Relational Token Lab — Experiment 3

## Learn the Composition Law

This is the next scientific step.

Experiments 0–2 gave the exact relational engine the composition law directly.

Experiment 3 does **not** give the learned relational model:

- complex multiplication,
- modular addition,
- a hard-coded Cayley table.

Instead it gets four learnable relation vectors and a neural binary operator:

```text
F(a, b) -> relation-state vector
```

The hidden synthetic world still uses the same four-relation law so the data generator
can create correct labels. The model itself does not receive that law.

## Three learned systems

### 1. Structured learned operator

Receives task examples plus structural constraints:

- relation 0 acts as identity;
- inverse pairings are supplied;
- associativity;
- closure onto the finite relation-state set;
- prototype separation.

Importantly, those constraints do not specify the full 4x4 composition table.

### 2. Unconstrained learned operator

Same neural relation embeddings and same binary operator, but no algebraic structural
regularizers.

### 3. Tiny transformer

Receives the same short sequences and final labels.

## Training and test

Train only on composition depths:

```text
1–5
```

Then test through:

```text
64
```

Afterward the two learned operators are evaluated in changing-topology graph episodes
without additional training.

## Run

Unzip this add-on into your existing `RelationalTokenLab` folder.

Activate the environment if needed:

```powershell
.venv\Scripts\activate
```

Then:

```powershell
python operator_app.py
```

Leave the defaults alone and click:

**Run learned-operator experiment**

Unlike Experiment 2, the progress bar now includes the post-training evaluation.

## Results

The app creates:

```text
operator_results/
    operator_depth_generalization.png
    operator_topology_relation.png
    operator_topology_coherence.png
    operator_depth_generalization.csv
    operator_topology_summary.csv
    learned_operator_tables.txt
    operator_report.json
    operator_models.pt
```

Send these four back to ChatGPT:

```text
operator_depth_generalization.png
operator_topology_relation.png
operator_topology_coherence.png
operator_report.json
```

The most important field in the report is:

```text
structured_operator.predicted_cayley_table
```

Compare it to:

```text
structured_operator.ground_truth_hidden_table
```

If they match, the learned neural operator reconstructed the hidden composition table
from shallow examples plus structural coherence constraints rather than being handed
the composition law.

## Interpretation limit

This remains a controlled synthetic world containing only four relation classes.
Success would show that a reusable composition law can be learned under strong
structural priors in this world. It would not establish automatic discovery of
arbitrary real-world mathematics or causality.
