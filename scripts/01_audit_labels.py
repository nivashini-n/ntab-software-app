"""Step 1 — annotation & header audit: dev pool + the 6 documented-defect subjects.

Writes results/audit_runs.csv, flags structural violations, renders gallery S1.
Holdout stays locked. Usage: uv run python scripts/01_audit_labels.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_decoding import config, labels, viz

RUNS = config.EXECUTED_RUNS + config.IMAGERY_RUNS

# ── Audit dev pool and (separately) the excluded subjects ────────────────────
dev = labels.audit_subjects(config.DEV, RUNS).assign(cohort="dev")
exc = labels.audit_subjects(sorted(config.EXCLUDED), RUNS).assign(cohort="excluded")
df = pd.concat([dev, exc], ignore_index=True)

config.RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(config.RESULTS_DIR / "audit_runs.csv", index=False)

# ── Report ───────────────────────────────────────────────────────────────────
flags_dev, flags_exc = labels.check(dev), labels.check(exc)
n_l, n_r = dev.n_T1.sum(), dev.n_T2.sum()
print(f"dev pool : {len(dev)} run files across {dev.subject.nunique()} subjects")
print(f"trials   : {n_l} left (T1) vs {n_r} right (T2)  → imbalance {n_l/(n_l+n_r):.1%} left")
print(f"structure: {len(flags_dev)} dev rows violate documented structure")
if len(flags_dev):
    print(flags_dev.to_string(index=False))
print(f"\nexcluded-subject verification ({len(flags_exc)}/{len(exc)} rows abnormal):")
cols = ["subject", "run", "sfreq", "dur_s", "n_T0", "n_T1", "n_T2", "dur_T1", "dur_T2"]
print(flags_exc[cols].to_string(index=False) if len(flags_exc) else "  none — defects NOT reproduced (investigate!)")

viz.fig_s1_label_audit(dev)
print("\nwrote results/audit_runs.csv + gallery stage S1")
