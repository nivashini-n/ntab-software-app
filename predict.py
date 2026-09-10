"""Grader-facing prediction CLI: raw EDF in → per-trial left/right labels out.

Usage (one line):
    uv run python predict.py path/to/S042R04.edf

Options:
    --model  models/ts_logreg_cross_subject.joblib   (default; trained on dev pool)
    --score  also compare predictions against the EDF's own T1/T2 annotations
    --json   write predictions to a JSON file instead of stdout

Contract: accepts any L/R-fist run (3,4,7,8,11,12) from any subject; applies the
IDENTICAL preprocessing module used in training (single source of truth); resamples
non-160 Hz files with a warning. T1/T2 are used only to locate trial onsets
(and as truth labels iff --score).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edf", type=Path, help="path to a raw EDF (L/R-fist run: R03/04/07/08/11/12)")
    p.add_argument("--model", type=Path, default=Path("models/ts_logreg_cross_subject.joblib"))
    p.add_argument("--score", action="store_true", help="compare against annotation-derived truth")
    p.add_argument("--json", type=Path, default=None, help="write predictions to this JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.edf.exists():
        sys.exit(f"EDF not found: {args.edf}")
    if not args.model.exists():
        sys.exit(f"Model not trained yet: {args.model} — run scripts/03_train_eval.py first (Phase B).")
    # TODO(Phase C): load model → preprocess.epochs pipeline → predict → print
    # per-trial: index, onset [s], predicted label (+ probability), and --score summary.
    raise NotImplementedError("TODO(Phase C): prediction path lands with the trained model.")


if __name__ == "__main__":
    main()
