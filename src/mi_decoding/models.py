"""The benchmark arms, as leakage-safe sklearn Pipelines.

Why Pipelines: CSP filters, tangent-space reference means, and classifier weights
are all FIT statistics — wrapping them guarantees they are learned inside CV
folds only. All arms end in linear classifiers on purpose: their evidence can be
drawn on a scalp map and checked against motor physiology (C3/C4).
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import config
from .features import laterality_features, trial_covariances

# ── Arm 1: physiological floor ───────────────────────────────────────────────
def make_laterality(ch_names: list[str]) -> Pipeline:
    """C3/C4 log band-power → logistic regression. Two features, one line of logic."""
    feat = FunctionTransformer(laterality_features, kw_args={"ch_names": list(ch_names)})
    return Pipeline([("lat", feat),
                     ("clf", LogisticRegression(max_iter=1000, random_state=config.SEED))])


# ── Arm 2: canonical baseline ────────────────────────────────────────────────
def make_csp_lda() -> Pipeline:
    """CSP(6, log-variance) → shrinkage LDA.

    CSP solves a generalized eigenproblem on the two class covariances: spatial
    filters whose output power is maximal for one class, minimal for the other.
    Its patterns_ plot as scalp maps — the built-in artifact lie-detector.
    """
    return Pipeline([("csp", CSP(n_components=config.N_CSP, reg="ledoit_wolf", log=True)),
                     ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


# ── Arms 3 & 4: Riemannian main model (+ unsupervised re-centering) ──────────
def make_ts_logreg() -> Pipeline:
    """Ledoit-Wolf covariance → tangent space at the training Riemannian mean →
    L2 logistic regression: a linear probe on a fixed geometric embedding."""
    return Pipeline([("cov", FunctionTransformer(trial_covariances)),
                     ("ts", TangentSpace(metric="riemann")),
                     ("clf", LogisticRegression(max_iter=2000, random_state=config.SEED))])


def fit_predict_recentered(covs_tr, y_tr, covs_te) -> np.ndarray:
    """Arm 4: train at the TRAIN mean, project test at the TEST subject's own mean
    (estimated from their UNLABELED covariances); classifier weights stay frozen.

    Not fine-tuning — no label-driven updates. It removes the per-person
    covariance offset, like batch-effect correction. Transductive in the
    unlabeled sense only (a real system would use a calibration recording).
    """
    ts_tr = TangentSpace(metric="riemann").fit(covs_tr)
    clf = LogisticRegression(max_iter=2000, random_state=config.SEED)
    clf.fit(ts_tr.transform(covs_tr), y_tr)
    ts_te = TangentSpace(metric="riemann").fit(covs_te)   # unlabeled re-centering
    return clf.predict(ts_te.transform(covs_te))
