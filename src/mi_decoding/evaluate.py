"""The evaluation ladder. Every reported number cites its rung.

Hypothesized shape: rung 0 ≥ rung 1 > rung 2 ≈ rung 3 (leakage → personalization →
new person). Treated as a hypothesis, not a script: whatever ordering the data
shows is what gets reported and explained (see DECISIONS.md pre-commitments).

Speed/correctness note: Ledoit-Wolf trial covariances are per-trial and stateless,
so they are precomputed ONCE and reused across folds — nothing is fit across
trials, hence no leakage. CSP/TangentSpace/classifiers are always fit per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from . import config

# ── Helpers ──────────────────────────────────────────────────────────────────
def _row(y_true, y_pred, **extra) -> dict:
    return {"acc": accuracy_score(y_true, y_pred),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "n": len(y_true), **extra}


def binom_ci95(p: float, n: int) -> float:
    """± half-width of the 95% normal-approx binomial CI."""
    return 1.96 * np.sqrt(p * (1 - p) / max(n, 1))


# ── Rung 0: pooled random split (DELIBERATELY leaky — demonstration only) ─────
def rung0_pooled(model, data, y) -> pd.DataFrame:
    """All subjects' epochs shuffled into stratified folds: same subject (and same
    run) on both sides of the split. Exists to show the inflated 'first number'."""
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    pred = cross_val_predict(clone(model), data, y, cv=cv)
    return pd.DataFrame([_row(y, pred, rung="0-pooled-leaky")])


# ── Rung 1: within-subject, grouped by run ───────────────────────────────────
def rung1_within_subject(model, data, y, subjects, runs) -> pd.DataFrame:
    """Personalized-BCI number: 3 folds = 3 runs, so no same-run leakage."""
    rows = []
    for s in np.unique(subjects):
        m = subjects == s
        cv = GroupKFold(min(3, len(np.unique(runs[m]))))  # tolerate a fully-rejected run
        pred = cross_val_predict(clone(model), data[m], y[m], cv=cv, groups=runs[m])
        rows.append(_row(y[m], pred, subject=s, rung="1-within-subject"))
    return pd.DataFrame(rows)


# ── Rung 2 (+2b): leave-one-subject-out, with shared training fits ───────────
def rung2_loso_ts(covs, y, subjects, progress=None) -> pd.DataFrame:
    """LOSO for the tangent-space model, standard (arm 3) and re-centered (arm 4)
    from ONE training fit per fold: train TS+LR once, then project the held-out
    subject at the train mean (standard) and at their own unlabeled mean
    (re-centered, frozen classifier)."""
    rows = []
    for i, s in enumerate(np.unique(subjects)):
        te, tr = subjects == s, subjects != s
        ts = TangentSpace(metric="riemann").fit(covs[tr])
        clf = LogisticRegression(max_iter=2000, random_state=config.SEED)
        clf.fit(ts.transform(covs[tr]), y[tr])
        pred_std = clf.predict(ts.transform(covs[te]))
        ts_re = TangentSpace(metric="riemann").fit(covs[te])      # unlabeled
        pred_re = clf.predict(ts_re.transform(covs[te]))
        rows += [_row(y[te], pred_std, subject=s, rung="2-loso", model="ts_logreg"),
                 _row(y[te], pred_re, subject=s, rung="2b-loso-recentered",
                      model="ts_logreg_recentered")]
        if progress and (i + 1) % 10 == 0:
            progress(f"    LOSO(ts) {i + 1} subjects done")
    return pd.DataFrame(rows)


def rung2_loso(model, data, y, subjects, name, progress=None) -> pd.DataFrame:
    """Generic LOSO (used for the laterality floor and CSP+LDA arms)."""
    rows = []
    for i, s in enumerate(np.unique(subjects)):
        te, tr = subjects == s, subjects != s
        m = clone(model).fit(data[tr], y[tr])
        rows.append(_row(y[te], m.predict(data[te]), subject=s, rung="2-loso", model=name))
        if progress and (i + 1) % 10 == 0:
            progress(f"    LOSO({name}) {i + 1} subjects done")
    return pd.DataFrame(rows)


# ── Rung 5: learned vs memorized ─────────────────────────────────────────────
def rung5_learning_curve(covs, y, subjects, sizes=(5, 10, 20, 40, 68),
                         n_eval=15, n_rep=3) -> pd.DataFrame:
    """LOSO-style accuracy vs number of training subjects, fixed 15-subject eval
    set, 3 seeded repeats — does more people help (learning) or not (memorizing)?"""
    rng = np.random.default_rng(config.SEED)
    subj = np.unique(subjects)
    eval_s = rng.choice(subj, n_eval, replace=False)
    pool = np.setdiff1d(subj, eval_s)
    rows = []
    for n in sizes:
        for rep in range(n_rep):
            tr_s = np.random.default_rng(config.SEED + 100 * rep + n).choice(pool, n, replace=False)
            tr = np.isin(subjects, tr_s)
            ts = TangentSpace(metric="riemann").fit(covs[tr])
            clf = LogisticRegression(max_iter=2000, random_state=config.SEED)
            clf.fit(ts.transform(covs[tr]), y[tr])
            for s in eval_s:
                te = subjects == s
                rows.append(_row(y[te], clf.predict(ts.transform(covs[te])),
                                 subject=s, n_train_subjects=n, rep=rep, rung="5-learning-curve"))
    return pd.DataFrame(rows)


def rung5_permutation_spotcheck(covs, y, subjects, runs, n_subj=10, n_perm=100) -> pd.DataFrame:
    """Label-shuffle null on 10 random subjects (within-subject CV): validates that
    the pipeline scores ~50% when the labels carry no information."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for s in rng.choice(np.unique(subjects), n_subj, replace=False):
        m = subjects == s
        ts = TangentSpace(metric="riemann")
        for p in range(n_perm):
            y_perm = rng.permutation(y[m])
            pred = np.empty_like(y_perm)
            for tr, te in GroupKFold(3).split(covs[m], y_perm, groups=runs[m]):
                t = ts.fit(covs[m][tr])
                clf = LogisticRegression(max_iter=2000).fit(t.transform(covs[m][tr]), y_perm[tr])
                pred[te] = clf.predict(t.transform(covs[m][te]))
            rows.append(_row(y_perm, pred, subject=s, perm=p, rung="5-permutation-null"))
    return pd.DataFrame(rows)


# ── Rung 8: executed → imagery transfer (within-subject, pre-committed) ──────
def rung8_transfer(covs_a, y_a, subj_a, covs_b, y_b, subj_b, direction) -> pd.DataFrame:
    """Train TS+LR on condition A's runs, test on condition B's runs, per subject."""
    rows = []
    for s in np.intersect1d(np.unique(subj_a), np.unique(subj_b)):
        tr, te = subj_a == s, subj_b == s
        ts = TangentSpace(metric="riemann").fit(covs_a[tr])
        clf = LogisticRegression(max_iter=2000, random_state=config.SEED)
        clf.fit(ts.transform(covs_a[tr]), y_a[tr])
        rows.append(_row(y_b[te], clf.predict(ts.transform(covs_b[te])),
                         subject=s, rung=f"8-transfer-{direction}"))
    return pd.DataFrame(rows)
