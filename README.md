# Left vs Right Fist Motor-Imagery Decoding — EEGMMIDB

NT@B FA26 Software Division take-home. Decodes *imagined* left- vs right-fist movement
from 64-channel EEG (PhysioNet EEGMMIDB, 109 subjects), with an evaluation designed to
answer the skeptical question first: **what else could explain this number?**

> Status: work in progress — see `DECISIONS.md` for the running engineering log.

## Quickstart

```bash
uv sync                                      # pinned environment
uv run python scripts/00_check_data.py       # verify dataset (set MI_DATA_DIR if needed)
# scripts 01→05 run the audit → preprocess → train/eval → holdout → figures
```

**Predict on a raw EDF (one line):**

```bash
uv run python predict.py data/S042/S042R04.edf
```

## Pipeline walkthrough (raw → result)

*Every processing stage below ships a figure; the full annotated gallery lives in
[`results/figures/GALLERY.md`](results/figures/GALLERY.md).*

- **S0 — What arrives:** 64 µV-scale channels @160 Hz + T0/T1/T2 event marks
- **S1 — Label audit:** we verified T1/T2 semantics and trial structure ourselves
- **S2 — Filtering:** 8–30 Hz (mu/beta) model view — where motor ERD lives
- **S3 — Epoching:** cue-locked 0.5–3.5 s windows
- **S4 — Rejection:** objective peak-to-peak threshold, impact quantified
- **S5 — What the model sees:** ERD maps, trial covariances, tangent-space embedding
- **S6 — What the model learned:** CSP topomaps, readout weights, per-subject accuracy
- **S7 — The evaluation story:** the leakage ladder, collapsed in one chart

## Models benchmarked

| Model | Role |
|---|---|
| C3/C4 mu-power laterality index | physiological floor (2 features) |
| CSP (6) + shrinkage LDA | canonical baseline |
| Riemannian tangent space + logistic regression | main model |
| + unsupervised per-subject re-centering | adaptation arm: frozen classifier, re-centered embedding |

## Evaluation ladder

Accuracy means nothing without its generalization claim. Every number is reported at its
rung: (0) pooled-leaky *(demonstration only)* → (1) within-subject → (2) leave-one-subject-out
→ (3) untouched 20-subject holdout, plus person-vs-task, memorization, and transfer analyses.

## Results

*(Phase B/C — tables + figures land here.)*

## Layout

```
src/mi_decoding/   config data labels preprocess features models evaluate analysis viz
scripts/           00_check_data … 05_figures (numbered run order)
predict.py         grader-facing CLI
results/figures/   the gallery (GALLERY.md = annotated index)
```
