"""Annotation audit — verify every label the models will train on.

Dataset docs define T0=rest and, in L/R-fist runs, T1=left / T2=right. Codes are
verified structurally here (counts, durations, rates); the semantic left/right
mapping is confirmed physiologically later (lateralized ERD over C3/C4).
"""

from __future__ import annotations

import mne
import pandas as pd

from . import config, data

# ── Documented run structure (what we verify against) ────────────────────────
EXPECT = {"sfreq": 160.0, "n_ch": 64, "n_task": 15, "task_dur": (3.9, 4.4)}


# ── Audit ────────────────────────────────────────────────────────────────────
def audit_run(subject: int, run: int, *, unlock_holdout: bool = False) -> dict:
    """Header + annotation summary for one run file (no preload — fast)."""
    config.assert_not_holdout([subject], unlock_holdout=unlock_holdout)
    raw = mne.io.read_raw_edf(data.edf_path(subject, run), preload=False, verbose="ERROR")
    ann, row = raw.annotations, {
        "subject": subject, "run": run,
        "sfreq": raw.info["sfreq"], "n_ch": len(raw.ch_names), "dur_s": round(float(raw.times[-1]), 1),
    }
    for code in ("T0", "T1", "T2"):
        m = ann.description == code
        row[f"n_{code}"] = int(m.sum())
        row[f"dur_{code}"] = round(float(ann.duration[m].mean()), 3) if m.any() else float("nan")
    return row


def audit_subjects(subjects, runs, *, unlock_holdout: bool = False) -> pd.DataFrame:
    """One audit row per subject × run."""
    rows = [audit_run(s, r, unlock_holdout=unlock_holdout) for s in subjects for r in runs]
    return pd.DataFrame(rows)


def check(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows violating the documented structure (empty = clean)."""
    lo, hi = EXPECT["task_dur"]
    bad = (
        (df.sfreq != EXPECT["sfreq"])
        | (df.n_ch != EXPECT["n_ch"])
        | (df.n_T1 + df.n_T2 != EXPECT["n_task"])
        | df.dur_T1.lt(lo) | df.dur_T1.gt(hi)
        | df.dur_T2.lt(lo) | df.dur_T2.gt(hi)
    )
    return df[bad]
