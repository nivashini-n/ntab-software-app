"""Step 4 — THE single-shot held-out test set evaluation (rung 3). Run ONCE.

Pre-committed protocol (DECISIONS.md, declared before this run): the headline arm
is the dev-LOSO winner — tangent space + logistic regression + unsupervised
re-centering. Plain TS+LR and CSP+LDA are reported as secondary, clearly labeled.
The only place in the codebase where unlock_holdout=True is passed.
Usage: uv run python scripts/04_final_holdout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
from pyriemann.tangentspace import TangentSpace

from mi_decoding import config, evaluate, models, preprocess
from mi_decoding.features import trial_covariances

# ── Load the frozen model and the held-out test subjects ─────────────────────
bundle = joblib.load(config.MODELS_DIR / "ts_logreg_cross_subject.joblib")
clf = bundle["pipeline"].named_steps["clf"]
ho = preprocess.build_dataset(list(config.HOLDOUT), config.IMAGERY_RUNS, unlock_holdout=True)
X, y, subj = ho["X"].astype(np.float64), ho["y"], ho["subject"]
covs = trial_covariances(X)
print(f"held-out test set: {len(config.HOLDOUT)} subjects, {len(y)} trials")

rows = []
# ── HEADLINE (pre-committed): frozen classifier + per-subject re-centering ────
for s in np.unique(subj):
    m = subj == s
    ts = TangentSpace(metric="riemann").fit(covs[m])           # unlabeled re-centering
    acc = (clf.predict(ts.transform(covs[m])) == y[m]).mean()
    rows.append({"subject": s, "model": "ts_logreg_recentered", "arm": "HEADLINE",
                 "acc": acc, "n": int(m.sum())})

# ── Secondary: frozen standard pipeline; CSP+LDA trained on dev ──────────────
pred_std = bundle["pipeline"].predict(X)
for s in np.unique(subj):
    m = subj == s
    rows.append({"subject": s, "model": "ts_logreg", "arm": "secondary",
                 "acc": (pred_std[m] == y[m]).mean(), "n": int(m.sum())})

dev = preprocess.build_dataset(list(config.DEV), config.IMAGERY_RUNS)
csp = models.make_csp_lda().fit(dev["X"].astype(np.float64), dev["y"])
pred_csp = csp.predict(X)
for s in np.unique(subj):
    m = subj == s
    rows.append({"subject": s, "model": "csp_lda", "arm": "secondary",
                 "acc": (pred_csp[m] == y[m]).mean(), "n": int(m.sum())})

df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / "holdout.csv", index=False)

print("\n=== HELD-OUT TEST SET (single shot, 20 subjects, imagery) ===")
for (arm, model), g in df.groupby(["arm", "model"]):
    pooled_correct = (g.acc * g.n).sum()
    pooled = pooled_correct / g.n.sum()
    ci = evaluate.binom_ci95(pooled, int(g.n.sum()))
    print(f"{arm:9s} {model:22s} mean-over-subjects {g.acc.mean():.3f} (sd {g.acc.std():.3f})"
          f"   pooled {pooled:.3f} ± {ci:.3f}")
print("\nwrote results/holdout.csv — these numbers are final and will not be re-run.")
