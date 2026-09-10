"""The evaluation ladder. Every reported number cites its rung.

Hypothesized shape: rung 0 ≥ rung 1 > rung 2 ≈ rung 3 (leakage → personalization →
new person). Treated as a hypothesis, not a script: whatever ordering the data
shows is what gets reported and explained (see DECISIONS.md pre-commitments).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# ── Rungs 0–3: the generalization ladder ─────────────────────────────────────
def rung0_pooled(X, y, subjects) -> pd.DataFrame:
    """DELIBERATELY LEAKY: epochs of all subjects shuffled into random folds.

    Exists to demonstrate the inflated 'first figure you see', never to report.
    Leaks subject identity + same-run context. TODO(Phase B).
    """
    raise NotImplementedError


def rung1_within_subject(X, y, subjects, runs) -> pd.DataFrame:
    """Per-subject GroupKFold over RUNS (3 folds) — the 'personalized BCI' number.

    Grouping by run kills same-run temporal leakage. TODO(Phase B).
    """
    raise NotImplementedError


def rung2_loso(X, y, subjects) -> pd.DataFrame:
    """Leave-one-subject-out on the dev pool — the 'new person walks in' number.

    TODO(Phase B).
    """
    raise NotImplementedError


def rung2b_loso_recentered(X, y, subjects) -> pd.DataFrame:
    """LOSO with unsupervised per-subject re-centering (the 4th benchmark arm).

    For each held-out subject: keep the trained classifier FROZEN, but re-estimate
    the tangent-space reference mean from their UNLABELED trial covariances and
    re-project there. No labels from the new person are ever used.
    Why: same move as batch-effect correction — remove the per-person covariance
    offset before applying a shared model. The rung2→rung2b delta MEASURES how much
    of the cross-subject gap is offset vs genuinely different neural patterns.
    TODO(Phase B).
    """
    raise NotImplementedError


def rung3_holdout(model, X_hold, y_hold) -> pd.DataFrame:
    """Single-shot locked-holdout evaluation (Phase C ONLY) + binomial CI.

    TODO(Phase C): called exclusively from scripts/04_final_holdout.py.
    """
    raise NotImplementedError


# ── Rungs 5, 6, 8: controls ──────────────────────────────────────────────────
def rung5_learning_curve(X, y, subjects) -> pd.DataFrame:
    """LOSO accuracy vs number of training subjects + train-test gap (memorized?).

    TODO(Phase B).
    """
    raise NotImplementedError


def rung5_permutation_null(X, y, subjects, n_perm: int = 200) -> pd.DataFrame:
    """Label-shuffle null — validates the significance machinery. TODO(Phase B)."""
    raise NotImplementedError


def rung6_baselines(X, y, subjects, ch_names) -> pd.DataFrame:
    """Chance, majority-class, C3/C4 laterality floor. TODO(Phase B)."""
    raise NotImplementedError


def rung8_transfer(X_exec, y_exec, X_imag, y_imag, subjects_exec, subjects_imag) -> pd.DataFrame:
    """Train executed → test imagery (and reverse), within-subject. TODO(Phase B)."""
    raise NotImplementedError
