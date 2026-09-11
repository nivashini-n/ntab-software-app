"""Single source of truth for every tunable in the pipeline.

Why: every choice lives in one place with its rationale (details in
DECISIONS.md), so any "why X?" has exactly one answer location.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MI_DATA_DIR", REPO_ROOT / "data"))  # PhysioNet layout Sxxx/SxxxRyy.edf
CACHE_DIR = REPO_ROOT / "cache"          # preprocessed epochs (gitignored)
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"    # demo-video gallery lives here
MODELS_DIR = REPO_ROOT / "models"

# ── Task definition ──────────────────────────────────────────────────────────
# L/R-fist runs ONLY — in fists/feet runs (5,6,9,10,13,14) T1/T2 mean other things.
# Semantics below are re-verified programmatically by scripts/01_audit_labels.py.
EXECUTED_RUNS = (3, 7, 11)               # real movement: stronger ERD, pipeline sanity check
IMAGERY_RUNS = (4, 8, 12)                # motor imagery: the headline task
EVENT_TO_LABEL = {"T1": "left", "T2": "right"}
CLASSES = ("left", "right")

# ── Subjects ─────────────────────────────────────────────────────────────────
# Excluded per documented defects (curation literature), re-verified in our audit:
# S089 wrong labels; S088/S092/S100 broken timing (S088 also 128 Hz); S038/S104 annotations.
EXCLUDED = frozenset({38, 88, 89, 92, 100, 104})
USABLE = tuple(s for s in range(1, 110) if s not in EXCLUDED)  # 103 subjects

# Locked holdout — generated ONCE via random.Random(26).sample(USABLE, 20), then
# hardcoded so it can never silently re-roll. Untouched until Phase C, enforced below.
HOLDOUT = (6, 8, 17, 22, 26, 27, 56, 57, 63, 66, 71, 78, 81, 83, 86, 91, 98, 101, 103, 106)
DEV = tuple(s for s in USABLE if s not in set(HOLDOUT))        # 83 subjects

# ── Preprocessing ────────────────────────────────────────────────────────────
SFREQ = 160.0                            # nominal; loader resamples + warns otherwise
BAND = (8.0, 30.0)                       # model view: mu+beta, where motor ERD lives
INSPECT_BAND = (1.0, 40.0)               # inspection view for PSD/TFR figures only
NOTCH_HZ = 60.0                          # US mains, applied to inspection view
EPOCH_TMIN, EPOCH_TMAX = -1.0, 4.0       # stored epoch (pre-cue kept for ERD baseline viz)
CROP = (0.5, 3.5)                        # analysis window: skip cue transient, stay in-trial
# Rejection revised 2026-09-10 from artifact statistics alone (never decoding results):
# 100 µV suited a single channel, but the statistic is the MAX over 64 channels, whose
# dev-pool median is 185 µV — it rejected 75% of epochs. See DECISIONS.md.
REJECT_P2P = 500e-6                      # reject epoch if any channel p2p exceeds (≈5% tail)
CH_FAULT_FRACTION = 0.5                  # channel alone rejecting >50% of a file → interpolate

# ── Models / evaluation ──────────────────────────────────────────────────────
N_CSP = 6                                # 3 filters per class
SEED = 26
EXEMPLAR_SUBJECT = 50                    # median rung-1 performer (64.4%) — chosen to avoid cherry-picking

# ── Holdout guard ────────────────────────────────────────────────────────────
def assert_not_holdout(subjects, *, unlock_holdout: bool = False) -> None:
    """Refuse to touch holdout subjects unless explicitly unlocked (Phase C only).

    Why: makes "we never looked at the holdout" an enforced property of the
    codebase rather than a promise.
    """
    if unlock_holdout:
        return
    leaked = sorted(set(subjects) & set(HOLDOUT))
    if leaked:
        raise RuntimeError(
            f"Holdout subjects {leaked} are locked until the final Phase C evaluation "
            "(scripts/04_final_holdout.py). Pass unlock_holdout=True only there."
        )


if __name__ == "__main__":  # `uv run python -m mi_decoding.config` → sanity summary
    found = sorted(p.name for p in DATA_DIR.glob("S[0-9][0-9][0-9]")) if DATA_DIR.exists() else []
    print(f"data dir     : {DATA_DIR}  (exists={DATA_DIR.exists()}, subject dirs={len(found)})")
    print(f"subjects     : usable={len(USABLE)}  dev={len(DEV)}  holdout={len(HOLDOUT)} (locked)")
    print(f"excluded     : {sorted(EXCLUDED)}")
    print(f"holdout      : {list(HOLDOUT)}")
    print(f"runs         : imagery={IMAGERY_RUNS}  executed={EXECUTED_RUNS}  labels={EVENT_TO_LABEL}")
    print(f"preprocess   : band={BAND} Hz  epoch=[{EPOCH_TMIN},{EPOCH_TMAX}]s  crop={CROP}s  "
          f"reject={REJECT_P2P*1e6:.0f} µV p2p")
