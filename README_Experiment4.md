# Relational Token Lab — Experiment 4

## Cross-Domain Coherence Transfer

This experiment is locked before the result.

### Core question
Can a relational state-update principle learned from simple toddler toy-taking / taking-back / permission / provenance cases provide measurable few-shot leverage in a semantically and institutionally different conversion-style legal domain?

### Domain A — toddlers
The surface act is usually the same: Child B takes a toy from Child A. The judgment changes with provenance: ownership, active permission, expired permission, authorization, or unresolved provenance. Irrelevant details are mixed into stories.

### Domain B — legal
The legal generator is intentionally more institutional: immediate possessory right, limited custody, delegated custody, authority chains, scope restrictions, revocation/expiration, downstream transfer, serious interference, delayed material facts, and irrelevant commercial facts. Legal training uses chain depths 1–2. The hard test uses depth 4.

This is a controlled experimental legal world, not a legal expert system and not legal advice. Its narrow rule core is designed around the familiar Arizona conversion concepts of dominion/control inconsistent with another's rights, a plaintiff's right to immediate possession at the relevant time, and serious interference.

### What transfers
After toddler training, only the learned relational state-update operator and initial latent state geometry are copied to the legal model. They are frozen.

The legal domain gets a completely fresh vocabulary, word embeddings/fact encoder, and classification head.

### Comparators
1. Transfer — intact toddler-pretrained frozen operator.
2. Scratch — identical relational architecture trained only on legal examples.
3. Scrambled — same transferred weight distribution/scale, but recurrence is surgically broken before freezing.
4. Tiny transformer — trained from scratch on the same labeled legal examples.

### Primary measurement
Legal-label budgets are locked at 16, 32, 64, 128, and 256 examples. Every system is evaluated on the same hard depth-4 legal test set.

### Special tests
- Material one-fact flip: one operative relation changes and the correct judgment must flip.
- Irrelevant perturbation: surface details change while operative relations stay fixed; judgment should remain stable.
- Delayed revelation: the initial record is indeterminate; a late material fact must cause a correct revised judgment.

### Locked criteria
At 64 legal examples:
- Transfer − Scratch >= 15 percentage points
- Transfer − Scrambled >= 15 percentage points
- Transfer − Transformer >= 10 percentage points

At 128 examples:
- Transfer hard-test accuracy >= 85%

At 256 examples:
- Material one-fact flip >= 85%
- Irrelevant surface invariance >= 90%
- Delayed revelation revision >= 80%

A mixed result stays mixed; do not collapse it to a boolean without reading the pattern.

## Run
Unzip into the same `RelationalTokenLab` folder used for the prior experiments.

If needed:
```powershell
.venv\Scripts\activate
```

Then:
```powershell
python crossdomain_app.py
```

Leave the defaults unchanged for the locked run.

## Send back
Upload:
- `crossdomain_results/crossdomain_sample_efficiency.png`
- `crossdomain_results/crossdomain_material_vs_irrelevant.png`
- `crossdomain_results/crossdomain_revelation_revision.png`
- `crossdomain_results/crossdomain_report.json`

`sample_cases.txt` is useful too if we want to inspect the generated stories.
