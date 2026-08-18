# Relational Reasoner v0.1

A deliberately **LLM-free** research library for building a persistent recursive relational reasoner with lexical retrieval.

The design goal is:

```text
question
  -> lexical retrieval (BM25)
  -> raw reference text
  -> trainable relational core
  -> native structured inference packet
  -> packet is re-fed as state
  -> repeat until ANSWER / UNKNOWN / fixed point
```

## Scientific boundary

This package intentionally does **not** use:

- large language models
- pretrained language encoders
- sentence-transformer embeddings
- vector databases that inject pretrained semantics
- domain-specific semantic lexicons
- hand-coded causal rules in the feedback controller

The package **does** provide a fixed baseline ontology and a trainable neural architecture. A target domain can be held out completely during training; at runtime, subject knowledge can come only from retrieved reference documents.

This code does **not** claim that arbitrary raw-text zero-shot reasoning is already solved. It gives you the architecture needed to test that claim without contaminating the result with another intelligent model.

## Baseline ontology

Primitive notions include entities, relations, properties, states, change, temporal order, comparisons, causation/enabling/prevention, provenance, and TRUE/FALSE/UNKNOWN.

The higher system-role layer uses:

```text
Boundary, Policy, Topology, Geometry, Level,
Rate, Constraint, Delay, Information
```

The ontology is domain-neutral. It does not contain facts such as `pressure -> geometry` or `jurisdiction -> boundary`.

## Key design rule

**Native model outputs are also valid model inputs.**

The feedback controller only appends the model's packet to the workspace and re-runs the same core. It contains no causal inference rules.

## Package layout

```text
relational_reasoner/
  ontology.py      fixed primitive/system ontology
  schema.py        propositions, packets, source references
  workspace.py     persistent state + provenance
  retrieval.py     pure lexical BM25 and document chunking
  tokenizer.py     byte/lexeme tokenization; no semantic vocabulary
  neural.py        trainable byte-level relational core
  training.py      JSONL training + losses
  controller.py    deterministic recursive feedback loop
  rag.py           retrieval + recursion orchestration
  render.py        deterministic structured output
  audit.py         contamination/audit manifest
  checkpoint.py    save/load helpers
  cli.py           index/train/ask commands
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

For PDF indexing:

```bash
pip install -e ".[pdf]"
```

## Index a reference library

```bash
relreason index --corpus ./references --out ./reference_index.json
```

Supported by default: `.txt`, `.md`. PDF works when the `pdf` extra is installed.

## Train the reasoning core

Training records are JSONL. See `examples/training_format.jsonl`.

```bash
relreason train --data examples/training_format.jsonl --out model.pt --steps 1000
```

For a real experiment, use a much larger generic relational/system training corpus and keep the target subject domain completely absent.

## Ask a question

```bash
relreason ask \
  --index ./reference_index.json \
  --checkpoint ./model.pt \
  "What causes cavitation when inlet pressure falls?"
```

The output is structured. Example shape:

```text
STATUS: ANSWER
TRUTH: TRUE
SUBJECT: inlet pressure
RELATION: contributes to
OBJECT: cavitation
ROLES: Geometry, State
SOURCES: pump_reference.pdf#chunk-17
STEPS: 4
```

The package does not add a prose-generating decoder.

## Zero-domain RAG protocol

The intended strong experiment is:

1. Train the relational core only on generic relational/system tasks.
2. Freeze **every parameter**.
3. Build a lexical index for a never-seen subject domain.
4. Ask questions about that subject.
5. Allow retrieved reference text to be the only new knowledge source.
6. Permit recursive re-feeding of the model's own inference packets.
7. Verify that the optimizer performs zero updates after the freeze point.

See `experiments/EXPERIMENT_6_PROTOCOL.md`.

## Important limitation

A baseline ontology is not the same thing as a semantic lexicon. The model still has to learn how arbitrary text expresses relational structure. If it cannot map unseen prose into its primitives, the experiment should fail rather than silently delegating that step to an LLM.
