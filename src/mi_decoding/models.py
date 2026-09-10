"""The two decoders + the floor baseline, as leakage-safe sklearn Pipelines.

Why Pipelines: CSP filters, covariance means, and scalers are all FIT statistics —
wrapping them guarantees they are learned inside CV folds only.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from . import config

# ── Decoders ─────────────────────────────────────────────────────────────────
def make_csp_lda() -> Pipeline:
    """Baseline: CSP(6) → log-variance → shrinkage LDA.

    Explainability hook: fitted CSP patterns plot as scalp maps that must
    localize over contralateral motor cortex (C3/C4) — else we decode artifact.
    TODO(Phase B): mne.decoding.CSP(n_components=config.N_CSP, log=True)
                   → LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").
    """
    raise NotImplementedError


def make_ts_logreg() -> Pipeline:
    """Main: LW covariance → Riemannian tangent space → L2 logistic regression.

    A linear probe on a geometry-respecting latent embedding; tangent projection
    re-centers each fit at its own Riemannian mean (helps cross-subject shift).
    TODO(Phase B): pyriemann Covariances("lwf") → TangentSpace() → LogisticRegression.
    """
    raise NotImplementedError


def make_laterality_baseline() -> Pipeline:
    """Floor: C3/C4 mu log-power → logistic regression (rung 6).

    TODO(Phase B).
    """
    raise NotImplementedError


def recenter_for_subject(fitted_ts_pipeline: Pipeline, X_new_covs) -> Pipeline:
    """Unsupervised adaptation: refit ONLY the tangent-space reference mean on the
    new subject's unlabeled covariances; classifier weights stay frozen.

    NOT fine-tuning (no pretrained weights, no label-driven updates) — a
    recentering of the embedding, analogous to per-site batch correction.
    TODO(Phase B): clone pipeline, refit TangentSpace reference, keep classifier.
    """
    raise NotImplementedError
