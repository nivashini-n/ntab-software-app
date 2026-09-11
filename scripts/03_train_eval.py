"""Step 3 — the full evaluation ladder on the DEV pool + gallery stages S5–S7.

Order: rung 1 (with executed-runs sanity gate) → rung 0 → identity probe →
LOSO (+re-centering) → learning curve → permutation null → transfer →
rejection sensitivity → save cross-subject model → tables → figures.
Writes results incrementally so partial runs still leave evidence.
Usage: uv run python scripts/03_train_eval.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace

from mi_decoding import analysis, config, data, evaluate, models, preprocess, viz
from mi_decoding.features import trial_covariances

T0 = time.time()
say = lambda msg: print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)
config.RESULTS_DIR.mkdir(exist_ok=True)
per_subject_frames = []

# ── Data ─────────────────────────────────────────────────────────────────────
im = preprocess.build_dataset(list(config.DEV), config.IMAGERY_RUNS)
ex = preprocess.build_dataset(list(config.DEV), config.EXECUTED_RUNS)
ch_names = data.load_raw(config.DEV[0], config.IMAGERY_RUNS[0]).ch_names
say(f"data: imagery {im['X'].shape}, executed {ex['X'].shape}")
covs_im = trial_covariances(im["X"])
covs_ex = trial_covariances(ex["X"])
say("trial covariances precomputed (per-trial + stateless → fold-safe)")

ts_pipe = lambda: Pipeline([("ts", TangentSpace(metric="riemann")),
                            ("clf", LogisticRegression(max_iter=2000, random_state=config.SEED))])

# ── Rung 1: within-subject (all arms, both conditions) ───────────────────────
r1 = {}
for task, dset, covs in (("imagery", im, covs_im), ("executed", ex, covs_ex)):
    for name, model, dat in (("laterality", models.make_laterality(ch_names), dset["X"]),
                             ("csp_lda", models.make_csp_lda(), dset["X"].astype(np.float64)),
                             ("ts_logreg", ts_pipe(), covs)):
        df = evaluate.rung1_within_subject(model, dat, dset["y"], dset["subject"], dset["run"])
        df["model"], df["task"] = name, task
        r1[(task, name)] = df
        per_subject_frames.append(df)
        say(f"rung1 {task:8s} {name:10s}: mean acc {df.acc.mean():.3f} (sd {df.acc.std():.3f})")

# SANITY GATE: executed must out-decode imagery for the main model, else STOP.
gap = r1[("executed", "ts_logreg")].acc.mean() - r1[("imagery", "ts_logreg")].acc.mean()
say(f"sanity gate: executed − imagery = {gap:+.3f} (expected > 0)")
if gap <= 0:
    sys.exit("OBSERVATION STOP: executed runs do not out-decode imagery — pipeline suspect.")

# ── Rung 0: pooled leaky demo (main model) ───────────────────────────────────
r0 = evaluate.rung0_pooled(ts_pipe(), covs_im, im["y"])
r0["model"], r0["task"] = "ts_logreg", "imagery"
say(f"rung0 pooled-leaky (demo): acc {r0.acc.iloc[0]:.3f}")

# ── Rung 4: identity vs class probe (same features, same split) ──────────────
probe = analysis.identity_probe(covs_im, im["y"], im["subject"], im["run"])
say(f"identity probe: subject-ID acc {probe['identity_acc']:.3f} "
    f"(chance {probe['identity_chance']:.3f}) vs class acc {probe['class_acc']:.3f} (chance 0.5)")
pd.DataFrame([probe]).to_csv(config.RESULTS_DIR / "identity_probe.csv", index=False)

# ── Rung 2 + 2b: LOSO (imagery) ──────────────────────────────────────────────
say("LOSO: tangent-space arms (83 folds, shared training fits) …")
r2_ts = evaluate.rung2_loso_ts(covs_im, im["y"], im["subject"], progress=say)
say("LOSO: csp_lda …")
r2_csp = evaluate.rung2_loso(models.make_csp_lda(), im["X"].astype(np.float64),
                             im["y"], im["subject"], "csp_lda", progress=say)
say("LOSO: laterality …")
r2_lat = evaluate.rung2_loso(models.make_laterality(ch_names), im["X"],
                             im["y"], im["subject"], "laterality", progress=say)
r2_all = pd.concat([r2_ts, r2_csp, r2_lat], ignore_index=True)
r2_all["task"] = "imagery"
per_subject_frames.append(r2_all)
for mod in r2_all.model.unique():
    sub = r2_all[r2_all.model == mod]
    for rung in sub.rung.unique():
        s2 = sub[sub.rung == rung]
        say(f"{rung} {mod}: mean acc {s2.acc.mean():.3f} (sd {s2.acc.std():.3f})")
pd.concat(per_subject_frames, ignore_index=True).to_csv(
    config.RESULTS_DIR / "per_subject.csv", index=False)

# ── Rung 5: learning curve + permutation null ────────────────────────────────
say("learning curve …")
lc = evaluate.rung5_learning_curve(covs_im, im["y"], im["subject"])
lc.to_csv(config.RESULTS_DIR / "learning_curve.csv", index=False)
say("learning curve: " + "  ".join(
    f"n={n}:{g.acc.mean():.3f}" for n, g in lc.groupby("n_train_subjects")))
say("permutation null (10 subjects × 100 shuffles) …")
perm = evaluate.rung5_permutation_spotcheck(covs_im, im["y"], im["subject"], im["run"])
perm.to_csv(config.RESULTS_DIR / "permutation_null.csv", index=False)
say(f"permutation null: mean {perm.acc.mean():.3f}, p95 {perm.acc.quantile(0.95):.3f}")

# ── Rung 8: transfer ─────────────────────────────────────────────────────────
r8_ei = evaluate.rung8_transfer(covs_ex, ex["y"], ex["subject"],
                                covs_im, im["y"], im["subject"], "exec-to-imag")
r8_ie = evaluate.rung8_transfer(covs_im, im["y"], im["subject"],
                                covs_ex, ex["y"], ex["subject"], "imag-to-exec")
pd.concat([r8_ei, r8_ie], ignore_index=True).to_csv(
    config.RESULTS_DIR / "transfer.csv", index=False)
say(f"transfer exec→imag: {r8_ei.acc.mean():.3f} (sd {r8_ei.acc.std():.3f}); "
    f"imag→exec: {r8_ie.acc.mean():.3f} (sd {r8_ie.acc.std():.3f})")

# ── Sensitivity: rejection on vs off (rung 1, main model) ────────────────────
say("sensitivity: rebuilding imagery without epoch rejection …")
im_nr = preprocess.build_dataset(list(config.DEV), config.IMAGERY_RUNS, reject=False)
covs_nr = trial_covariances(im_nr["X"])
r1_nr = evaluate.rung1_within_subject(ts_pipe(), covs_nr, im_nr["y"],
                                      im_nr["subject"], im_nr["run"])
say(f"sensitivity: rung1 ts_logreg no-rejection {r1_nr.acc.mean():.3f} "
    f"vs with-rejection {r1[('imagery', 'ts_logreg')].acc.mean():.3f}")
r1_nr.assign(model="ts_logreg", task="imagery-norej").to_csv(
    config.RESULTS_DIR / "sensitivity_norej.csv", index=False)

# ── Ladder summary table ─────────────────────────────────────────────────────
rows = [{"rung": "0-pooled-leaky", "model": "ts_logreg", "acc": r0.acc.iloc[0],
         "ci": evaluate.binom_ci95(r0.acc.iloc[0], int(r0.n.iloc[0]))}]
for name in ("laterality", "csp_lda", "ts_logreg"):
    df = r1[("imagery", name)]
    rows.append({"rung": "1-within-subject", "model": name, "acc": df.acc.mean(),
                 "ci": 1.96 * df.acc.std() / np.sqrt(len(df))})
for (rung, mod), g in r2_all.groupby(["rung", "model"]):
    rows.append({"rung": rung, "model": mod, "acc": g.acc.mean(),
                 "ci": 1.96 * g.acc.std() / np.sqrt(len(g))})
ladder = pd.DataFrame(rows)
ladder.to_csv(config.RESULTS_DIR / "ladder.csv", index=False)
say("ladder table written:\n" + ladder.to_string(index=False))

# ── Cross-subject model for predict.py (trained on ALL dev imagery) ──────────
final = models.make_ts_logreg()
final.fit(im["X"].astype(np.float64), im["y"])
config.MODELS_DIR.mkdir(exist_ok=True)
joblib.dump({"pipeline": final, "ch_names": ch_names, "classes": list(config.CLASSES),
             "band": config.BAND, "crop": config.CROP, "epoch": (config.EPOCH_TMIN, config.EPOCH_TMAX),
             "trained_on": f"{len(config.DEV)} dev subjects, imagery runs {config.IMAGERY_RUNS}",
             "eval": "LOSO mean acc: see results/ladder.csv"},
            config.MODELS_DIR / "ts_logreg_cross_subject.joblib", compress=3)
say("saved models/ts_logreg_cross_subject.joblib")

# ── Exemplar subject: median rung-1 imagery performer ────────────────────────
r1_ts = r1[("imagery", "ts_logreg")].sort_values("acc").reset_index(drop=True)
exemplar = int(r1_ts.subject.iloc[len(r1_ts) // 2])
say(f"exemplar (median rung-1) subject: S{exemplar:03d} acc {r1_ts.acc.iloc[len(r1_ts) // 2]:.3f}")

# ── Gallery stages S5–S7 ─────────────────────────────────────────────────────
say("figures: S5 ERD grand average (20 subjects, inspection view) …")
rng = np.random.default_rng(config.SEED)
erd_subj = sorted(rng.choice(config.DEV, 20, replace=False))
eps = []
for s in erd_subj:
    for r in config.IMAGERY_RUNS:
        iv = preprocess.inspect_view(data.load_raw(s, r))
        eps.append(preprocess.epochs_from_raw(iv, r).pick(["C3", "C4"]))
ep_all = mne.concatenate_epochs(eps, verbose="ERROR")
viz.fig_s5_erd(ep_all["left"], ep_all["right"], len(erd_subj))

viz.fig_s5_covariances(covs_im, im["y"], ch_names)

say("figures: t-SNE embedding of tangent vectors …")
Z = TangentSpace(metric="riemann").fit(covs_im).transform(covs_im)
xy = TSNE(n_components=2, perplexity=30, init="pca",
          random_state=config.SEED).fit_transform(PCA(50, random_state=config.SEED).fit_transform(Z))
viz.fig_s5_embedding(xy, im["y"], im["subject"])

say("figures: S6 …")
info = preprocess.model_view(data.load_raw(config.DEV[0], 4)).info
csp_viz = models.make_csp_lda().named_steps["csp"]
csp_viz.fit(im["X"].astype(np.float64), im["y"])
viz.fig_s6_csp_patterns(csp_viz, info)

W = np.abs(final.named_steps["clf"].coef_[0])
iu = np.triu_indices(64)
M = np.zeros((64, 64)); M[iu] = W; M = M + M.T
viz.fig_s6_readout(M.sum(0), info)

chance_half = 1.96 * np.sqrt(0.25 / r1[("imagery", "ts_logreg")].n.median())
viz.fig_s6_per_subject(r1[("imagery", "ts_logreg")],
                       r2_all[(r2_all.model == "ts_logreg") & (r2_all.rung == "2-loso")],
                       (0.5 - chance_half, 0.5 + chance_half))

viz.fig_s7_ladder(ladder)
viz.fig_s7_transfer(r8_ei, r8_ie, r1[("imagery", "ts_logreg")].acc.mean())
say("gallery: S5, S6, S7 written. DONE.")
