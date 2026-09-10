"""Step 2 — filter → epoch → reject → cache the dev pool; gallery stages S0, S2–S4.

Builds both task datasets (imagery + executed), reports drop/QC stats, and renders
the raw/filtering/epoching/rejection figures from the exemplar subject.
Usage: uv run python scripts/02_preprocess_cache.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mne

from mi_decoding import config, data, preprocess, viz

# ── Build + cache both task datasets ─────────────────────────────────────────
datasets = {}
for task, runs in (("imagery", config.IMAGERY_RUNS), ("executed", config.EXECUTED_RUNS)):
    t0 = time.time()
    ds = datasets[task] = preprocess.build_dataset(list(config.DEV), runs)
    n_total = sum(m["n_total"] for m in ds["meta"])
    rep = [(m["subject"], m["run"], m["interp"]) for m in ds["meta"] if m["interp"]]
    print(f"{task:8s}: X{ds['X'].shape}  kept {len(ds['y'])}/{n_total} "
          f"({100 * (n_total - len(ds['y'])) / n_total:.2f}% dropped)  [{time.time() - t0:.0f}s]")
    print(f"          repaired electrodes: {rep if rep else 'none'}")

# ── Gallery: S0 raw, S2 filtering, S3 epoching, S4 rejection ────────────────
s = config.EXEMPLAR_SUBJECT
raw = data.load_raw(s, config.IMAGERY_RUNS[0])
mv = preprocess.model_view(raw)
viz.fig_s0_raw(raw, s)
viz.fig_s0_montage(raw)
viz.fig_s2_filtering(raw, mv)

ep_full = mne.concatenate_epochs(  # all 3 imagery runs → enough trials to stack
    [preprocess.epochs_from_raw(preprocess.model_view(data.load_raw(s, r)), r)
     for r in config.IMAGERY_RUNS], verbose="ERROR")
viz.fig_s3_epoching(mv, ep_full, s)
viz.fig_s4_rejection(datasets["imagery"]["meta"])

print("gallery: wrote stages S0, S2, S3, S4 → results/figures/ (see GALLERY.md)")
