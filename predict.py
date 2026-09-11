"""Grader-facing prediction CLI: raw EDF in → per-trial left/right labels out.

Usage (one line):
    uv run python predict.py path/to/S042R04.edf

Options:
    --model    models/ts_logreg_cross_subject.joblib (default; trained on 83 dev subjects)
    --score    also compare predictions against the EDF's own T1/T2 annotations
    --recenter re-center the tangent-space reference on this file's unlabeled trials
               (unsupervised adaptation; classifier weights unchanged)
    --json     write predictions to a JSON file as well

Contract: accepts any left/right-fist run (R03/04/07/08/11/12) from any EEGMMIDB
subject. Applies the IDENTICAL preprocessing used in training (same module —
single source of truth): channel standardization, 8–30 Hz band-pass, cue-locked
0.5–3.5 s epochs, electrode repair. Non-160 Hz files are resampled with a
warning. No epoch is dropped — every cue gets a prediction. T1/T2 annotations
are used only to locate trial onsets (and as truth labels iff --score).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edf", type=Path, help="raw EDF path (L/R-fist run: R03/04/07/08/11/12)")
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "models/ts_logreg_cross_subject.joblib")
    p.add_argument("--score", action="store_true", help="compare against annotation-derived truth")
    p.add_argument("--recenter", action="store_true", help="unsupervised per-file re-centering")
    p.add_argument("--json", type=Path, default=None, help="also write predictions to this path")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.edf.exists():
        sys.exit(f"EDF not found: {args.edf}")
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model} — run scripts/03_train_eval.py first.")

    import joblib
    import mne
    import numpy as np

    mne.set_log_level("ERROR")
    from mi_decoding import data, preprocess
    from mi_decoding.features import trial_covariances

    bundle = joblib.load(args.model)
    pipe, classes = bundle["pipeline"], bundle["classes"]

    # ── Load with the training-time standardization; sanity-warn on run type ──
    raw = data.load_raw_any(args.edf)
    m = re.search(r"R(\d{2})", args.edf.name)
    run = int(m.group(1)) if m else None
    if run in (1, 2):
        sys.exit(f"{args.edf.name} looks like a baseline run (no T1/T2 cues to predict).")
    if run in (5, 6, 9, 10, 13, 14):
        print(f"WARNING: {args.edf.name} looks like a fists/feet run — T1/T2 do not mean "
              "left/right fist there; predictions below assume the L/R-fist task.")

    missing = [ch for ch in bundle["ch_names"] if ch not in raw.ch_names]
    if missing:
        sys.exit(f"EDF lacks expected channels: {missing[:6]} …")
    raw.reorder_channels(bundle["ch_names"])

    # ── Identical preprocessing path; repair but never drop (every cue answered) ──
    mv = preprocess.model_view(raw)
    ep = preprocess.epochs_from_raw(mv, run=None).crop(*bundle["crop"])
    ep, p2p, interp = preprocess.repair_and_reject(ep, reject=False)
    if interp:
        print(f"note: repaired persistently-noisy channel(s): {interp}")

    X = ep.get_data(copy=True).astype(np.float64)
    if args.recenter:
        from pyriemann.tangentspace import TangentSpace
        covs = trial_covariances(X)
        ts = TangentSpace(metric="riemann").fit(covs)          # this file's own mean
        z = ts.transform(covs)
        clf = pipe.named_steps["clf"]
        pred_i, proba = clf.predict(z), clf.predict_proba(z)
    else:
        pred_i, proba = pipe.predict(X), pipe.predict_proba(X)
    preds = [classes[i] for i in pred_i]

    # ── Report ────────────────────────────────────────────────────────────────
    onsets = ep.events[:, 0] / mv.info["sfreq"]                # cue sample → seconds
    flagged = p2p > 500e-6
    print(f"\n{args.edf.name}: {len(preds)} trials "
          f"({'re-centered' if args.recenter else 'standard'} tangent-space model)")
    for k, (t, lab, pr) in enumerate(zip(onsets, preds, proba.max(1))):
        star = "  [high-amplitude epoch]" if flagged[k] else ""
        print(f"  trial {k:2d}  cue at {t:7.2f}s  →  {lab:5s}  (p={pr:.2f}){star}")

    out = [{"trial": int(k), "onset_s": float(t), "pred": lab, "p": float(pr)}
           for k, (t, lab, pr) in enumerate(zip(onsets, preds, proba.max(1)))]
    if args.score:
        truth = [classes[c - 1] for c in ep.events[:, 2]]
        acc = float(np.mean([a == b for a, b in zip(preds, truth)]))
        print(f"\n--score vs annotations: {acc:.3f}  ({sum(a == b for a, b in zip(preds, truth))}"
              f"/{len(truth)} trials)")
        for o, tr in zip(out, truth):
            o["truth"] = tr
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
