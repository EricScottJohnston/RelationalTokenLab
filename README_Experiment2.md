# Relational Token Lab — Experiment 2

## Changing Relational Topology / Boundary

This experiment isolates the next hypothesis:

> If the computational state is the relational structure itself, then adding or
> removing relationships should immediately change what is inferable, without
> retraining the system.

Experiment 1 already tested the controlled language-to-relation mapping. Experiment 2
therefore removes language on purpose and gives all systems the same relation IDs. This
lets us test **topology change** rather than conflate it with language learning.

## Install

If you already ran Experiment 1, you already have the required packages.

Unzip the Experiment 2 add-on into the same `RelationalTokenLab` folder.

Activate your existing environment if necessary:

```powershell
.venv\Scripts\activate
```

Run:

```powershell
python topology_app.py
```

Leave the default settings alone for the first run and click:

**Run topology experiment**

## What gets trained

Two learned baselines:

- a tiny transformer over graph edge records;
- a small real-valued GNN with four message-passing rounds.

They see only static graphs with **4–8 nodes**.

The exact relational engine has no training. It is the explicit Z4 relational algebra.

## What gets tested

Without retraining, test graphs use:

```text
8, 12, 16, 20, 24 nodes
```

Every test episode changes topology:

```text
BASE
  coherent graph, source and target connected

CUT
  remove one relationship on their path
  -> source and target become disconnected
  -> correct relation becomes UNKNOWN

RECONNECT
  add a correct cross-component relation
  -> source and target become related again

CONTRADICTION
  add a wrong edge
  -> graph becomes incoherent

REPAIR
  remove the wrong edge
  -> coherence returns
```

## Outputs

The app creates:

```text
topology_results/
    topology_relation_generalization.png
    topology_coherence_generalization.png
    topology_event_sequence.png
    topology_event_results.csv
    topology_size_summary.csv
    topology_episode.txt
    topology_report.json
    topology_models.pt
```

Send these four files back to ChatGPT:

```text
topology_relation_generalization.png
topology_coherence_generalization.png
topology_event_sequence.png
topology_report.json
```

## What a positive result means

A strong result would show that the exact relational system remains correct as the
graph changes and grows, because inference is recomputed from the current relations
rather than learned as a fixed-size pattern.

The transformer and GNN are intentionally meaningful baselines:

- the transformer must statistically learn graph inference from edge records;
- the GNN is graph-native, so this is not merely "graphs beat sequences."

If both learned baselines also generalize perfectly, the test is too easy and should
be hardened.

If the exact relational engine remains exact while learned baselines degrade with
graph size or changing boundaries, that is evidence for the structural inductive bias
we are testing.

It is still a synthetic Z4 world. It does not by itself establish general-purpose
reasoning, unrestricted causal inference, or natural-language superiority.
