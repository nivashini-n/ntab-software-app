"""Rung 4 (person vs task) + rung 7 (interpretability) analyses.

Why: the scope's sharpest questions — 'how much is the person, not the task?'
and 'why does the model say what it says?' — are answered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Rung 4: person vs task ───────────────────────────────────────────────────
def subject_identity_probe(X, subjects) -> pd.DataFrame:
    """Train a classifier to predict SUBJECT ID from the same features.

    High identity-decodability quantifies how much 'person' lives in the
    representation the L/R model uses (same confound logic as leave-one-site-out
    batch effects in multi-site imaging). TODO(Phase B).
    """
    raise NotImplementedError


def per_subject_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy distribution across subjects + binomial CIs (BCI illiteracy).

    TODO(Phase B).
    """
    raise NotImplementedError


# ── Rung 7: interpretability ─────────────────────────────────────────────────
def erd_time_frequency(epochs_by_class) -> dict:
    """Morlet TFR contrast at C3 vs C4 for left vs right — the physiological signal.

    TODO(Phase B): baseline-corrected (pre-cue only) ERD/ERS maps.
    """
    raise NotImplementedError


def tangent_embedding(X, y, subjects) -> np.ndarray:
    """2-D t-SNE of tangent-space vectors, plotted colored by class AND by subject.

    Expectation: subject clusters dominate class structure — the geometric
    picture behind the rung 0 → rung 2 accuracy collapse.
    (t-SNE over umap-learn: identical interpretive value, no llvmlite build
    dependency — umap-learn failed to build on Intel macOS.) TODO(Phase B).
    """
    raise NotImplementedError
