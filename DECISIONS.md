# Engineering Log

Chronological record of design decisions, alternatives considered, attempts, and outcomes.
Entries are written when the decision is made; outcomes are appended when known.

---

## 2026-09-09 — Task framing: imagery-first, executed as control, plus transfer
**Decision:** headline task = left vs right fist on the *imagined* runs (4, 8, 12); the
*executed* runs (3, 7, 11) serve as a stronger-signal control; add an executed→imagery
transfer analysis.
**Alternatives:** imagery-only; executed-only.
**Rationale:** imagery is the BCI-relevant problem. Executed movement produces larger
event-related desynchronization (ERD), so it bounds the pipeline: if execution can't be
decoded, imagery results aren't trustworthy. Transfer asks whether the two tasks share a
neural representation.
**Caveat (added 09-10):** in executed runs, muscle/movement artifacts correlate perfectly
with class — EMG bleeds into the beta band, so 8–30 Hz filtering attenuates but does not
eliminate it. Executed results are reported as an upper bound, possibly EMG-assisted;
imagery is free of this confound by design. CSP pattern topographies (central vs peripheral
foci) are the check.

## 2026-09-09 — Models: two linear pipelines + a physiological floor
**Decision:** (a) CSP (6 components) + shrinkage LDA; (b) Ledoit-Wolf trial covariance →
Riemannian tangent space → L2 logistic regression; (c) floor baseline: C3/C4 mu log-power
laterality index.
**Alternatives:** CSP-only; adding a compact CNN (EEGNet).
**Rationale:** with ~22 trials per class per subject, small-sample-efficient linear methods
are the right operating point, and both are inspectable — CSP patterns plot as scalp maps
(they should localize over contralateral motor cortex, C3/C4; peripheral foci would indicate
artifact decoding), and the tangent-space model is a linear readout on a fixed geometric
embedding of the trial covariance. A deep model at this trial count mostly risks fitting
subject idiosyncrasies; deferred to future work.

## 2026-09-09 — Subjects: all usable (103), with a 20-subject locked holdout
**Decision:** exclude the 6 subjects with documented defects (S038, S088, S089, S092, S100,
S104 — wrong labels, broken trial timing, or 128 Hz sampling; re-verified in our own audit,
see 09-10 audit entry). Of the 103 usable, lock 20 subjects (drawn once with seed 26,
hardcoded in `config.py`) that no analysis touches until the single final evaluation.
The loader enforces this with a runtime guard (`assert_not_holdout`).
**Alternatives:** fewer subjects for faster iteration; no holdout (LOSO only).
**Rationale:** cross-subject generalization is the question of interest; a never-touched
subject set is the only split that can't be contaminated by iteration. Compute is not a
constraint for linear models.

## 2026-09-09 — Noise handling: band-pass + fixed-threshold rejection; no ICA
**Decision:** models see 8–30 Hz band-passed data (mu+beta, where motor ERD lives); a
separate 1–40 Hz + 60 Hz notch view exists for inspection figures only. Epochs exceeding
100 µV peak-to-peak (a priori threshold) are rejected; % dropped is reported per subject,
with a with/without-rejection sensitivity check.
**Alternative seriously considered:** ICA with automated component labeling.
**Rationale:** the band-pass already removes the dominant artifact energy (drift <1 Hz,
blinks <4 Hz, line 60 Hz); a fixed threshold is objective (no per-component judgment calls)
and cheap. Accepted cost: residual in-band EMG — quantified by the sensitivity check rather
than assumed away.

## 2026-09-10 — Data acquisition
physionet.org served an expired TLS certificate, so the dataset (3.4 GB, 109 subjects) was
downloaded manually and the repo reads it through a gitignored `data/` symlink;
`MI_DATA_DIR` overrides the location. All 109 subject folders verified present, 0 missing
run files; smoke test on S001R04: 160 Hz, 64 channels, annotations 15×T0 / 8×T1 / 7×T2 —
matching the documented ~15-trial run structure, including the mild class imbalance
(hence balanced accuracy will be reported alongside accuracy).

## 2026-09-10 — Stage-figure gallery
**Decision:** every pipeline stage produces a before/after figure (S0 raw → S7 evaluation),
indexed with captions in `results/figures/GALLERY.md`, all regenerable from scripts.
**Rationale:** the processing chain should be visible, not asserted — a reader can see what
the models are trained on and what they learned at each transformation.

## 2026-09-10 — Benchmark arms
**Decision:** four arms, all evaluated on identical splits: (1) C3/C4 laterality floor,
(2) CSP+LDA, (3) tangent space + logistic regression, (4) arm 3 + unsupervised per-subject
re-centering — for a new subject, the tangent-space reference mean is re-estimated from
their *unlabeled* trial covariances; classifier weights stay frozen.
**Terminology note:** arm 4 is not fine-tuning — no pretrained weights and no label-driven
updates; it recenters the embedding only.
**Disclosure:** re-centering is transductive in the unlabeled sense (it uses the test
subject's unlabeled data, as a real calibration recording would). Stated explicitly so the
evaluation assumptions are auditable. With only ~22–45 trials the mean estimate is noisy;
the arm-3 → arm-4 delta is reported whether positive, null, or negative.

## 2026-09-10 — Attempt: umap-learn → switched to scikit-learn t-SNE
Tried `umap-learn` for the tangent-space embedding figure; its `llvmlite` dependency has no
x86_64-macOS wheel and failed building from source. Switched to scikit-learn's t-SNE: same
interpretive value (nonlinear neighborhood-preserving 2-D projection), one fewer fragile
dependency in the pinned environment.

## 2026-09-10 — Environment notes
uv-managed environment, exact versions pinned via `uv.lock` (mne 1.13.0, pyriemann 0.12,
scikit-learn 1.9.1, Python 3.11). MNE 1.13 renamed the `standard_1005` montage to
`colin27_1005`; code uses the new name and the mne pin is `>=1.13,<2` accordingly.

## 2026-09-10 — Evaluation protocol pre-commitments
Recorded before any model result exists, so measurement choices cannot chase outcomes:

1. **Ladder as hypothesis, not script.** Expected ordering is pooled-leaky ≥ within-subject >
   cross-subject; whatever ordering the data actually shows is what gets reported and
   explained. (With linear models, pooling 83 heterogeneous subjects can cost more than
   leakage gains — if rung 1 exceeds rung 0, that is the finding.)
2. **Holdout unlock protocol.** The headline arm for the single-shot holdout evaluation is
   whichever arm wins dev-pool LOSO, *declared in this log before unlocking*. Other arms may
   be run on the holdout afterward but are reported as secondary, clearly labeled. No
   model selection happens on holdout results.
3. **Subject-identity probe split.** The probe (predicting subject ID from the same features)
   uses run-grouped splits — train on two runs, test on the held-out run — so it measures
   identity, not shared run context.
4. **Fixed a-priori knobs.** Epoch analysis window (0.5–3.5 s post-cue) and rejection
   threshold (100 µV p2p) were set from literature/structure before any decoding result;
   both get sensitivity reports rather than tuning.
5. **Per-channel QC.** Channels that are flat (<0.01 µV std) or extreme (>10× median std) on
   the model view are detected and logged during preprocessing; whether to interpolate is
   decided at the observation checkpoint, not silently.
6. **Objective documentation.** Negative results, dead ends, and deviations from expectation
   are logged here as findings, not smoothed over.

## 2026-09-10 — Audit results (Phase A observation)
Dev pool 100% structurally clean: 498 run files, 83 subjects, 0 violations; 3,756 left vs
3,714 right trials (50.3/49.7). Excluded-subject defects reproduced independently for 5 of 6:
S088/S092/S100 recorded at 128 Hz with 5.125 s trials (wrong paradigm timing throughout),
S089 run 3 structurally deviant (181 s, 22 rest periods), S104 run 8 truncated (106 s,
13 trials, 3.94 s durations). **S038 passed every structural check** — its documented defect
is not visible in headers or annotations. Decision: keep it excluded; the defect may be
semantic (wrong label assignments, invisible to structural checks — S089's other five runs
also look clean, and its known defect IS semantic). Reported as an audit finding.

## 2026-09-10 — Rejection threshold miscalibration (found and fixed)
First pass rejected **75% of epochs** (161/249 files lost everything). Diagnosis: 100 µV was
calibrated to single-channel amplitude, but the rejection statistic is the MAX over 64
channels — on band-passed dev data that statistic's median is 185 µV (p99 = 877 µV); in-band
alpha/beta bursts at any single electrode cross 100 µV routinely. Same diagnostics exposed
extreme subject heterogeneity (file-median p2p from 25 µV to 1,755 µV) and persistent
electrode faults (S109's C4 above threshold in 5/6 runs; one screaming channel on S079).
**Fix, from artifact statistics only (no decoding result existed yet):** two-level cleaning —
(1) repair: a channel that alone would reject >50% of a file's epochs, or is flat, is an
electrode fault → interpolated from neighbors (MNE spherical splines); (2) reject: epochs
whose remaining max-channel p2p exceeds a fixed 500 µV (≈ the 5% tail). This supersedes the
earlier flat/10×-median QC flag rule — detection and repair are now one mechanism. Everything
uses stock MNE operations (`interpolate_bads`, p2p rejection); the only custom logic is the
6-line fault-detection rule. The with/without-rejection sensitivity check stays.

## Open items
- Exemplar subject for single-subject figures (S001 provisional; revisit after audit).
- Interpolation policy if per-channel QC flags anything (decide on observation).
- Final choice of the "beyond the prompts" analysis (candidates: identity dominance in the
  embedding, resting mu power vs decodability, audit irregularities).
