"""Continuous EDF → clean, epoch-level model input. Every step has a figure (viz.py).

Pipeline: filter (model/inspection views) → cue-locked epochs → p2p rejection.
Why no ICA: a fixed objective threshold has no subjective component-selection
step; the 8–30 Hz band already removes dominant artifact energy (drift <1 Hz,
blinks <4 Hz, 60 Hz line). Residual in-band EMG is the accepted, quantified cost.
Why no re-reference: CSP/covariance features are invariant to invertible linear
spatial transforms; CAR would also reduce matrix rank.
"""

from __future__ import annotations

import joblib
import mne
import numpy as np

from . import config, data

# ── Filtering: two views ─────────────────────────────────────────────────────
def model_view(raw: mne.io.Raw) -> mne.io.Raw:
    """8–30 Hz band-pass (mu+beta). The ONLY view models ever see."""
    return raw.copy().filter(*config.BAND, fir_design="firwin", verbose="ERROR")


def inspect_view(raw: mne.io.Raw) -> mne.io.Raw:
    """1–40 Hz + 60 Hz notch. For inspection figures only — never for models."""
    return (raw.copy()
            .notch_filter(config.NOTCH_HZ, verbose="ERROR")
            .filter(*config.INSPECT_BAND, fir_design="firwin", verbose="ERROR"))


# ── Two-level cleaning: repair electrode faults, reject transient bursts ─────
def channel_faults(epochs: mne.Epochs) -> list[str]:
    """Channels that are flat, or alone exceed REJECT_P2P in >50% of epochs.

    Why: a channel that is artifactual for most of a run is an electrode fault —
    repair it by interpolation instead of discarding the subject's trials.
    """
    d = epochs.get_data(copy=False)
    p2p_ch = d.max(axis=2) - d.min(axis=2)                    # (n_epochs, n_ch)
    frac = (p2p_ch > config.REJECT_P2P).mean(axis=0)
    flat = d.std(axis=2).mean(axis=0) < 1e-8
    return [ch for ch, f, fl in zip(epochs.ch_names, frac, flat)
            if fl or f > config.CH_FAULT_FRACTION]


def repair_and_reject(epochs: mne.Epochs, *, reject: bool = True
                      ) -> tuple[mne.Epochs, np.ndarray, list[str]]:
    """Interpolate persistent electrode faults, then (optionally) drop burst epochs."""
    bads = channel_faults(epochs)
    if bads:
        epochs = epochs.copy()
        epochs.info["bads"] = bads
        epochs.interpolate_bads(verbose="ERROR")
    p2p = peak_to_peak(epochs)
    kept = epochs[p2p <= config.REJECT_P2P] if reject else epochs
    return kept, p2p, bads


# ── Epoching & rejection ─────────────────────────────────────────────────────
def epochs_from_raw(raw: mne.io.Raw, run: int | None) -> mne.Epochs:
    """Cue-locked epochs [-1, 4] s. Labels valid ONLY in L/R-fist runs.

    run=None (predict path, unknown filename): skip the whitelist and trust the
    caller to have warned about T1/T2 semantics.
    """
    if run is not None and run not in config.EXECUTED_RUNS + config.IMAGERY_RUNS:
        raise ValueError(f"run {run}: T1/T2 do not mean left/right fist in this run")
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
    return mne.Epochs(raw, events, event_id={"left": 1, "right": 2},
                      tmin=config.EPOCH_TMIN, tmax=config.EPOCH_TMAX,
                      baseline=None, preload=True, verbose="ERROR")


def peak_to_peak(epochs: mne.Epochs) -> np.ndarray:
    """Per-epoch peak-to-peak amplitude, max over channels (volts)."""
    d = epochs.get_data(copy=False)
    return (d.max(axis=2) - d.min(axis=2)).max(axis=1)


def reject_epochs(epochs: mne.Epochs) -> tuple[mne.Epochs, np.ndarray]:
    """Drop epochs exceeding REJECT_P2P on the analysis window; return p2p values."""
    p2p = peak_to_peak(epochs)
    return epochs[p2p <= config.REJECT_P2P], p2p


# ── Dataset assembly ─────────────────────────────────────────────────────────
def build_dataset(subjects: list[int], runs: tuple[int, ...], *, unlock_holdout: bool = False,
                  use_cache: bool = True, reject: bool = True) -> dict:
    """Assemble {X, y, subject, run, meta} on the CROP window.

    X: (n_trials, 64, n_times) float32; y: 0=left, 1=right. meta holds per-file
    drop stats, pre-rejection p2p values, and repaired channels. reject=False
    keeps electrode repair but skips epoch rejection (sensitivity check).
    """
    config.assert_not_holdout(subjects, unlock_holdout=unlock_holdout)
    tag = (f"r{'-'.join(map(str, runs))}_n{len(subjects)}"
           f"_b{config.BAND[0]:g}-{config.BAND[1]:g}_c{config.CROP[0]:g}-{config.CROP[1]:g}"
           f"_rej{config.REJECT_P2P * 1e6:g}_rep{config.CH_FAULT_FRACTION:g}"
           + ("" if reject else "_norej"))
    cache = config.CACHE_DIR / f"epochs_{tag}.joblib"
    if use_cache and cache.exists():
        return joblib.load(cache)

    X, y, subj, run_id, meta = [], [], [], [], []
    for i, s in enumerate(subjects):
        for r in runs:
            mv = model_view(data.load_raw(s, r, unlock_holdout=unlock_holdout))
            ep = epochs_from_raw(mv, r).crop(*config.CROP)
            kept, p2p, interp = repair_and_reject(ep, reject=reject)
            X.append(kept.get_data(copy=True).astype(np.float32))
            y.append(kept.events[:, 2] - 1)
            subj.append(np.full(len(kept), s))
            run_id.append(np.full(len(kept), r))
            meta.append({"subject": s, "run": r, "n_total": len(ep), "n_kept": len(kept),
                         "p2p": p2p, "interp": interp})
        if (i + 1) % 20 == 0:
            print(f"  … {i + 1}/{len(subjects)} subjects")

    out = {"X": np.concatenate(X), "y": np.concatenate(y),
           "subject": np.concatenate(subj), "run": np.concatenate(run_id), "meta": meta}
    config.CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(out, cache, compress=3)
    return out
