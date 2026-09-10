"""The pipeline visualization gallery — one stage, one figure, one caption.

Why: the submission walks reviewers through the data raw→result instead of
jumping to numbers. Every figure lands in results/figures/ AND is indexed with
its caption in results/figures/GALLERY.md — the demo-video companion doc.
Caption formula: what you're seeing → what changed → why we did it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, not windows
import matplotlib.pyplot as plt

from . import config

# ── Style: consistent, colorblind-safe, presentation-ready ───────────────────
CLASS_COLORS = {"left": "#4C72B0", "right": "#DD8452"}  # fixed across ALL figures
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

_GALLERY = config.FIGURES_DIR / "GALLERY.md"


def save_fig(fig: plt.Figure, name: str, caption: str) -> None:
    """Save a stage figure and register it in the gallery index.

    `name` convention: 's{stage}_{slug}' → results/figures/s2_psd_before_after.png.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    header = "# Pipeline gallery — walkthrough for the demo video\n\nRegenerate: `uv run python scripts/05_figures.py`\n"
    lines = _GALLERY.read_text().splitlines() if _GALLERY.exists() else header.splitlines()
    lines = [l for l in lines if f"]({name}.png)" not in l and not l.startswith(f"### {name}")]
    lines += [f"### {name}", f"![{name}]({name}.png)", "", caption, ""]
    _GALLERY.write_text("\n".join(lines) + "\n")


# ── Stage figures (implemented alongside their pipeline stage) ───────────────
def fig_s0_raw(raw, subject: int):
    """10-s raw trace (~9 chans) with T0/T1/T2 spans + montage with C3/Cz/C4. TODO(Phase A)."""
    raise NotImplementedError


def fig_s1_label_audit(audit: "pd.DataFrame"):
    """Event counts per run/class, duration histogram, class balance. TODO(Phase A)."""
    raise NotImplementedError


def fig_s2_filtering(raw_before, raw_after):
    """PSD before/after (mu/beta shaded, 60 Hz visible) + trace before/after. TODO(Phase A)."""
    raise NotImplementedError


def fig_s3_epoching(raw, epochs):
    """Continuous→epochs schematic + trials×time heatmap at C3 per class. TODO(Phase A)."""
    raise NotImplementedError


def fig_s4_rejection(epochs, drop_stats: dict):
    """Kept vs rejected epoch with threshold drawn + %% dropped per subject. TODO(Phase A)."""
    raise NotImplementedError


def fig_s5_model_input(epochs_by_class, covs, y, tangent_xy=None):
    """ERD/ERS TFR at C3 vs C4; class-mean covariances + diff; tangent t-SNE. TODO(Phase B)."""
    raise NotImplementedError


def fig_s6_model_learned(csp, lda, info, per_subject_acc):
    """CSP pattern topomaps; readout weights at channel level; accuracy distribution. TODO(Phase B)."""
    raise NotImplementedError


def fig_s7_evaluation(ladder: "pd.DataFrame", transfer: "pd.DataFrame"):
    """The ladder-collapse chart (rungs 0→3) + transfer bars. TODO(Phase B)."""
    raise NotImplementedError
