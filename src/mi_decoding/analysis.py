"""Rung 4 (person vs task) + supporting analyses.

Why: the sharpest skeptical questions — 'how much is the person, not the task?'
and 'did it learn or memorize?' — are answered here, on the same features the
decoder uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from . import config

# ── Rung 4: subject-identity probe ───────────────────────────────────────────
def identity_probe(covs, y, subjects, runs, train_runs=(4, 8), test_run=12) -> dict:
    """Predict SUBJECT ID (83 classes) vs predict CLASS (2) from identical features
    and an identical run-grouped split (train runs 4+8 → test run 12).

    Run-grouping matters: same-run splits would measure shared run context, not
    identity. The identity/class contrast quantifies how much 'who is recording'
    dominates 'what they imagined' in the representation.
    """
    tr, te = np.isin(runs, train_runs), runs == test_run
    ts = TangentSpace(metric="riemann").fit(covs[tr])          # fit on train runs only
    Ztr, Zte = ts.transform(covs[tr]), ts.transform(covs[te])

    id_clf = LogisticRegression(max_iter=3000, random_state=config.SEED)
    id_clf.fit(Ztr, subjects[tr])
    id_acc = (id_clf.predict(Zte) == subjects[te]).mean()

    cls_clf = LogisticRegression(max_iter=2000, random_state=config.SEED)
    cls_clf.fit(Ztr, y[tr])
    cls_acc = (cls_clf.predict(Zte) == y[te]).mean()

    n_subj = len(np.unique(subjects))
    return {"identity_acc": id_acc, "identity_chance": 1 / n_subj,
            "class_acc": cls_acc, "class_chance": 0.5, "n_test": int(te.sum())}


# ── Per-subject summaries (BCI illiteracy) ───────────────────────────────────
def per_subject_summary(df: pd.DataFrame) -> dict:
    """Distribution stats + how many subjects sit inside the chance CI."""
    accs, n_med = df.acc.values, int(df.n.median())
    half = 1.96 * np.sqrt(0.25 / n_med)                        # 95% CI around 0.5
    return {"mean": accs.mean(), "sd": accs.std(), "min": accs.min(), "max": accs.max(),
            "n_median_trials": n_med, "chance_band": (0.5 - half, 0.5 + half),
            "n_at_chance": int((accs <= 0.5 + half).sum()), "n_subjects": len(accs)}
