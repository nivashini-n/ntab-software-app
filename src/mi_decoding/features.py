"""Trial representations the models consume.

Two paths from the same 8–30 Hz epochs: a supervised low-dim compression (CSP,
lives inside models.py) and the full spatial covariance per trial (Riemannian
path). Plus the 2-feature physiological floor: C3/C4 log band-power.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances

# ── Riemannian path ──────────────────────────────────────────────────────────
def trial_covariances(X: np.ndarray) -> np.ndarray:
    """(n, 64, t) → (n, 64, 64) Ledoit-Wolf covariances.

    Why shrinkage: ~480 samples for a 64×64 matrix is borderline; LW guarantees
    well-conditioned SPD matrices with an analytically chosen intensity.
    Per-trial and stateless → safe to precompute once outside CV folds.
    """
    return Covariances(estimator="lwf").transform(X.astype(np.float64))


# ── Physiological floor features ─────────────────────────────────────────────
def laterality_features(X: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Log band-power at C3 and C4 only — the 2-feature minimum model (rung 6).

    Why: if the real models barely beat two electrodes and a linear rule,
    honesty requires saying so.
    """
    idx = [ch_names.index("C3"), ch_names.index("C4")]
    return np.log(X[:, idx, :].var(axis=2))
