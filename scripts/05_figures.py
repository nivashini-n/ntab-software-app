"""Step 5 — regenerate the ENTIRE gallery (S0–S7) + GALLERY.md from caches/results.

One command rebuilds every figure in results/figures/. Requires scripts 01–04 to
have run (uses their caches and CSVs). Runtime ≈ 5 min (ERD grand average and the
t-SNE dominate). Usage: uv run python scripts/05_figures.py
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mi_decoding import config, data, evaluate, models, preprocess, viz
from mi_decoding.features import trial_covariances

say = lambda m: print(m, flush=True)
gallery = config.FIGURES_DIR / "GALLERY.md"
if gallery.exists():
    gallery.unlink()                     # rebuild the index from scratch, in stage order

# ── Shared inputs ─────────────────────────────────────────────────────────────
im = preprocess.build_dataset(list(config.DEV), config.IMAGERY_RUNS)
covs_im = trial_covariances(im["X"])
audit = pd.read_csv(config.RESULTS_DIR / "audit_runs.csv")
per_subj = pd.read_csv(config.RESULTS_DIR / "per_subject.csv")
ladder = pd.read_csv(config.RESULTS_DIR / "ladder.csv")
transfer = pd.read_csv(config.RESULTS_DIR / "transfer.csv")
bundle = joblib.load(config.MODELS_DIR / "ts_logreg_cross_subject.joblib")
s = config.EXEMPLAR_SUBJECT

# ── S0–S3: exemplar-subject stages ────────────────────────────────────────────
say(f"S0–S3 (exemplar S{s:03d}) …")
raw = data.load_raw(s, config.IMAGERY_RUNS[0])
mv = preprocess.model_view(raw)
viz.fig_s0_raw(raw, s)
viz.fig_s0_montage(raw)
viz.fig_s1_label_audit(audit[audit.cohort == "dev"])
viz.fig_s2_filtering(raw, mv)
ep_full = mne.concatenate_epochs(
    [preprocess.epochs_from_raw(preprocess.model_view(data.load_raw(s, r)), r)
     for r in config.IMAGERY_RUNS], verbose="ERROR")
viz.fig_s3_epoching(mv, ep_full, s)
viz.fig_s4_rejection(im["meta"])

# ── S5: ERD grand average, covariances, embedding ────────────────────────────
say("S5 ERD grand average (all dev subjects) …")
eps = [preprocess.epochs_from_raw(preprocess.inspect_view(data.load_raw(d, r)), r).pick(["C3", "C4"])
       for d in config.DEV for r in config.IMAGERY_RUNS]
ep_all = mne.concatenate_epochs(eps, verbose="ERROR")
viz.fig_s5_erd(ep_all["left"], ep_all["right"], len(config.DEV))
viz.fig_s5_covariances(covs_im, im["y"], bundle["ch_names"])

say("S5 t-SNE embedding …")
Z = TangentSpace(metric="riemann").fit(covs_im).transform(covs_im)
xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=config.SEED
          ).fit_transform(PCA(50, random_state=config.SEED).fit_transform(Z))
viz.fig_s5_embedding(xy, im["y"], im["subject"])

# ── S6: what the models learned ───────────────────────────────────────────────
say("S6 …")
info = mv.info
csp = models.make_csp_lda().named_steps["csp"]
csp.fit(im["X"].astype(np.float64), im["y"])
viz.fig_s6_csp_patterns(csp, info)
W = np.abs(bundle["pipeline"].named_steps["clf"].coef_[0])
iu = np.triu_indices(64)
M = np.zeros((64, 64)); M[iu] = W; M = M + M.T
viz.fig_s6_readout(M.sum(0), info)
r1 = per_subj[(per_subj.rung == "1-within-subject") & (per_subj.model == "ts_logreg")
              & (per_subj.task == "imagery")]
r2 = per_subj[(per_subj.rung == "2-loso") & (per_subj.model == "ts_logreg")]
half = 1.96 * np.sqrt(0.25 / r1.n.median())
viz.fig_s6_per_subject(r1, r2, (0.5 - half, 0.5 + half))

# ── S7: the evaluation story ──────────────────────────────────────────────────
say("S7 …")
viz.fig_s7_ladder(ladder)
viz.fig_s7_transfer(transfer[transfer.rung == "8-transfer-exec-to-imag"],
                    transfer[transfer.rung == "8-transfer-imag-to-exec"],
                    r1.acc.mean(), r1.acc.std())
say("gallery rebuilt → results/figures/ (index: GALLERY.md)")
