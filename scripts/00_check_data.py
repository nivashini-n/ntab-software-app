"""Step 0 — verify the dataset is present, complete, and parseable.

Prints: subjects found, missing runs, and a one-run smoke test (sampling rate,
channels, annotation counts) proving the EDF+ → MNE path end-to-end.
Usage: uv run python scripts/00_check_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_decoding import config, data

# ── Inventory ────────────────────────────────────────────────────────────────
subjects = data.available_subjects()
print(f"data dir : {config.DATA_DIR}")
print(f"subjects : {len(subjects)} found (expect 109); usable per config: {len(config.USABLE)}")

missing = [(s, r) for s in subjects for r in range(1, 15) if not data.edf_path(s, r).exists()]
print(f"missing  : {len(missing)} run files" + (f" → {missing[:5]}..." if missing else ""))

# ── One-run smoke test (dev-pool subject only — holdout guard stays active) ──
s, r = config.EXEMPLAR_SUBJECT, config.IMAGERY_RUNS[0]
raw = data.load_raw(s, r)
codes = sorted(set(raw.annotations.description))
counts = {c: int((raw.annotations.description == c).sum()) for c in codes}
print(f"\nsmoke    : S{s:03d}R{r:02d} — {raw.info['sfreq']:.0f} Hz, {len(raw.ch_names)} ch, "
      f"{raw.times[-1]:.1f} s")
print(f"annotations: {counts}   (T1=left fist, T2=right fist in this run — audited in step 01)")
