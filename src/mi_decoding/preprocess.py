"""Continuous EDF → clean, epoch-level model input. Every step has a figure (viz.py).

Pipeline: filter (model/inspection views) → cue-locked epochs → p2p rejection.
Why no ICA: a fixed objective threshold has no subjective component-selection
step; the 8–30 Hz band already removes dominant artifact energy (drift <1 Hz,
blinks <4 Hz, 60 Hz line). Residual in-band EMG is the accepted, quantified cost.
Why no re-reference: CSP/covariance features are invariant to invertible linear
spatial transforms; CAR would also reduce matrix rank.
"""

from __future__ import annotations

import mne
import numpy as np

from . import config

# ── Filtering: two views ─────────────────────────────────────────────────────
def model_view(raw: mne.io.Raw) -> mne.io.Raw:
    """8–30 Hz band-pass (mu+beta). The ONLY view models ever see.

    TODO(Phase A): FIR band-pass, return filtered copy.
    """
    raise NotImplementedError


def inspect_view(raw: mne.io.Raw) -> mne.io.Raw:
    """1–40 Hz + 60 Hz notch. For PSD/ERD/TFR figures only — never for models.

    TODO(Phase A).
    """
    raise NotImplementedError


# ── Epoching & rejection ─────────────────────────────────────────────────────
def epochs_from_raw(raw: mne.io.Raw, run: int) -> mne.Epochs:
    """Cue-locked epochs [-1, 4] s from T1/T2 annotations, labels via config map.

    Run must be whitelisted (L/R-fist runs only) — T1/T2 mean other things elsewhere.
    TODO(Phase A).
    """
    raise NotImplementedError


def reject_epochs(epochs: mne.Epochs) -> tuple[mne.Epochs, dict]:
    """Drop epochs exceeding REJECT_P2P peak-to-peak; return (kept, drop_stats).

    Why fixed a-priori threshold: nothing tuned on test data; % dropped is reported
    per subject and a with/without sensitivity check ships with results.
    TODO(Phase A).
    """
    raise NotImplementedError


# ── Dataset assembly ─────────────────────────────────────────────────────────
def build_dataset(
    subjects: list[int], runs: tuple[int, ...], *, unlock_holdout: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X, y, subject_ids, run_ids) for the analysis window CROP.

    X: (n_trials, 64, n_times) float64. groups arrays drive leakage-safe CV.
    Caches to CACHE_DIR keyed by preprocessing params.
    TODO(Phase A).
    """
    raise NotImplementedError
