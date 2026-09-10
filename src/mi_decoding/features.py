"""Trial representations the models consume.

Two paths from the same 8–30 Hz epochs:
  CSP path        — supervised 6-dim compression of covariance structure (in models.py).
  Riemannian path — full 64×64 Ledoit-Wolf covariance per trial → tangent space
                    ("embed each trial in a principled latent space, fit a linear probe").
Plus the minimal physiological baseline: C3/C4 mu log-power laterality.
"""

from __future__ import annotations

import numpy as np

# ── Riemannian path ──────────────────────────────────────────────────────────
def trial_covariances(X: np.ndarray) -> np.ndarray:
    """(n_trials, 64, n_times) → (n_trials, 64, 64) Ledoit-Wolf covariances.

    Why shrinkage: ~480 samples for a 64×64 matrix is borderline; LW guarantees
    well-conditioned SPD matrices with an analytic (tune-free) intensity.
    TODO(Phase B): pyriemann.estimation.Covariances(estimator="lwf").
    """
    raise NotImplementedError


# ── Minimal baseline features ────────────────────────────────────────────────
def laterality_index(X: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """2-feature physiological floor: log mu-power at C3 and C4 per trial.

    Why: if the real models barely beat this two-electrode index, honesty
    demands saying so (evaluation ladder rung 6).
    TODO(Phase B).
    """
    raise NotImplementedError
