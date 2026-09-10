"""Annotation audit — the scope says "check T1/T2 yourself", so we do, in code.

Why: every label the models train on is verified here first; the audit also
re-confirms the documented defects behind the 6 excluded subjects.
"""

from __future__ import annotations

import mne
import pandas as pd

# ── Per-run audit ────────────────────────────────────────────────────────────
def audit_run(raw: mne.io.Raw, subject: int, run: int) -> pd.DataFrame:
    """One row per annotation: onset, duration, code — plus run metadata.

    TODO(Phase A): extract raw.annotations into a tidy frame.
    """
    raise NotImplementedError


def audit_subjects(subjects: list[int], runs: tuple[int, ...]) -> pd.DataFrame:
    """Full audit table: event counts/durations/class balance per subject × run.

    Checks (Phase A):
      1. Only T0/T1/T2 codes appear; counts per run ≈ 15 task trials (~7–8/class).
      2. Task durations ≈ 4.1–4.2 s; flag deviants (catches the S088/S092/S100 defects).
      3. Class balance per run → justifies also reporting balanced accuracy.
      4. Sampling rate == 160 Hz everywhere (catches S088's 128 Hz).
    TODO(Phase A): implement + write results/audit.csv and a summary print.
    """
    raise NotImplementedError
