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
_TRACE_CHANS = ["Fp1", "Fz", "C3", "Cz", "C4", "P3", "P4", "O1", "O2"]


def _stacked_traces(ax, times, data_uv, chans, spacing=120):
    """µV traces stacked with fixed offsets; returns the offsets used."""
    import numpy as np
    off = np.arange(len(chans))[::-1] * spacing
    for i in range(len(chans)):
        ax.plot(times, data_uv[i] + off[i], lw=0.6, color="#333")
    ax.set_yticks(off, chans)
    return off


def _shade_cues(ax, raw, t0, t1):
    """Shade T1/T2 cue spans inside [t0, t1] with the class colors."""
    for on, du, de in zip(raw.annotations.onset, raw.annotations.duration,
                          raw.annotations.description):
        if de in ("T1", "T2") and t0 <= on <= t1:
            c = CLASS_COLORS["left" if de == "T1" else "right"]
            ax.axvspan(on, min(on + du, t1), color=c, alpha=0.18, lw=0)


def fig_s0_raw(raw, subject: int):
    """10 s of raw signal, 9 channels, with the left/right cue spans shaded."""
    import numpy as np
    start = float(raw.annotations.onset[1]) - 2  # around the first task cues
    sl = raw.copy().pick(_TRACE_CHANS).crop(start, start + 20)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _stacked_traces(ax, sl.times + start, sl.get_data() * 1e6, _TRACE_CHANS, spacing=150)
    _shade_cues(ax, raw, start, start + 20)
    ax.set(xlabel="time (s)", title=f"S{subject:03d} — raw EEG as it arrives (µV, unfiltered)")
    save_fig(fig, "s0_raw_trace",
             "**S0 — What arrives.** Twenty seconds of raw EEG from 9 of the 64 channels. "
             "Shaded spans are the cue annotations embedded in the EDF (blue = left-fist cue, "
             "orange = right-fist cue; unshaded gaps = rest). Note the scale: tens of µV, "
             "dominated by slow drift and broadband noise — no class difference is visible "
             "to the eye. Everything downstream exists to change that.")


def fig_s0_montage(raw):
    """All 64 sensor positions; the sensorimotor row highlighted."""
    import numpy as np
    pos = raw.get_montage().get_positions()["ch_pos"]
    xy = np.array([pos[ch][:2] for ch in raw.ch_names])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(plt.Circle((0, 0), 0.095, fill=False, lw=1))
    ax.plot([-0.016, 0, 0.016], [0.0935, 0.106, 0.0935], color="k", lw=1)  # nose
    ax.scatter(*xy.T, s=14, color="#bbb", zorder=2)
    # contralateral teaching cue: C3 (left hemi) ↔ RIGHT hand, C4 ↔ LEFT hand
    for ch, c in (("C3", CLASS_COLORS["right"]), ("Cz", "#555"), ("C4", CLASS_COLORS["left"])):
        x, y = pos[ch][:2]
        ax.scatter(x, y, s=48, color=c, zorder=3)
        ax.annotate(ch, (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("64-channel montage (10-10 system)")
    save_fig(fig, "s0_montage",
             "**S0 — Where the signal should live.** Sensor layout over the scalp (nose up). "
             "C3 sits over left motor cortex and C4 over right motor cortex. Because motor "
             "control is contralateral, imagining the LEFT hand should suppress rhythms at "
             "C4 (orange ↔ left class) and the RIGHT hand at C3 (blue ↔ right class) — the "
             "spatial hypothesis every later figure gets checked against.")


def fig_s1_label_audit(audit):
    """Trial counts per run, task-duration histogram, per-subject class balance."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    per_run = audit.groupby("run")[["n_T1", "n_T2"]].mean()
    per_run.plot.bar(ax=axes[0], color=[CLASS_COLORS["left"], CLASS_COLORS["right"]], rot=0,
                     legend=False)
    axes[0].set(title="Mean trials per run", ylabel="trials", xlabel="run")

    import pandas as pd
    durs = pd.concat([audit.dur_T1.rename("left"), audit.dur_T2.rename("right")], axis=1)
    counts = durs.round(3).apply(pd.Series.value_counts).fillna(0)
    counts.plot.bar(ax=axes[1], color=[CLASS_COLORS["left"], CLASS_COLORS["right"]], rot=0)
    axes[1].set(title="Task-segment duration (discrete)", xlabel="seconds", ylabel="run files")
    axes[1].legend(["left (T1)", "right (T2)"], frameon=False)

    per_subj = audit.groupby("subject")[["n_T1", "n_T2"]].sum()
    axes[2].scatter(per_subj.n_T1, per_subj.n_T2, s=14, alpha=0.6, color="#555")
    axes[2].axline((22, 22), slope=1, ls="--", lw=0.8, color="gray")
    axes[2].set(title="Per-subject trial totals", xlabel="left trials", ylabel="right trials")

    save_fig(fig, "s1_label_audit",
             "**S1 — Label audit.** Left: every L/R-fist run carries ~15 cued trials with a "
             "~8/7 class split (the imbalance is why balanced accuracy is reported too). "
             "Middle: task segments take exactly two durations, 4.088 or 4.100 s — a BCI2000 "
             "timing artifact, uniform across the pool. Right: per-subject class totals lie on "
             "an anti-diagonal (left + right ≈ 90 fixed trials), with the split ranging only "
             "42–48 — mild, bounded imbalance and no class-skewed subjects.")


def fig_s2_filtering(raw, raw_mv):
    """PSD raw vs model view + a C3 trace before/after the 8–30 Hz band-pass."""
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
    for r, lab, c in ((raw, "raw", "#aaa"), (raw_mv, "model view (8–30 Hz)", "#333")):
        psd = r.compute_psd(fmax=80, verbose="ERROR")
        axes[0].semilogy(psd.freqs, psd.get_data().mean(0), lw=1.1, label=lab, color=c)
    axes[0].axvspan(8, 13, color=CLASS_COLORS["left"], alpha=0.10)
    axes[0].axvspan(13, 30, color=CLASS_COLORS["right"], alpha=0.08)
    axes[0].axvline(60, ls=":", lw=0.9, color="r")
    axes[0].annotate("60 Hz line", (60, axes[0].get_ylim()[1] * 0.4), color="r",
                     fontsize=8, rotation=90, va="top")
    axes[0].annotate("mu", (9.5, axes[0].get_ylim()[1] * 0.45), fontsize=9)
    axes[0].annotate("beta", (19, axes[0].get_ylim()[1] * 0.45), fontsize=9)
    axes[0].set(xlabel="frequency (Hz)", ylabel="power (V²/Hz)", title="Channel-average PSD")
    axes[0].legend(frameon=False, fontsize=8)

    start = float(raw.annotations.onset[1]) - 2
    for r, lab, off in ((raw, "raw", 90), (raw_mv, "8–30 Hz", 0)):
        seg = r.copy().pick(["C3"]).crop(start, start + 10)
        axes[1].plot(seg.times + start, seg.get_data()[0] * 1e6 + off, lw=0.7,
                     color="#333" if off == 0 else "#aaa", label=lab)
    axes[1].set(xlabel="time (s)", ylabel="µV (offset)", title="C3, same 10 s, before/after")
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "s2_filtering",
             "**S2 — Filtering.** Left: the average power spectrum. Raw EEG (grey) shows the "
             "1/f drift ramp at low frequencies and the 60 Hz mains line. The 8–30 Hz "
             "band-pass (black) keeps exactly the mu (8–13 Hz) and beta (13–30 Hz) bands "
             "where motor imagery suppresses rhythmic power (ERD), and removes drift, blink "
             "energy (<4 Hz), most EMG, and line noise by construction. Right: the same C3 "
             "segment before/after — what remains is the band-limited oscillation whose "
             "per-trial power the models consume.")


def fig_s3_epoching(raw_mv, epochs_full, subject: int):
    """Continuous→epochs windowing on real data + per-class trials×time image at C3."""
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), width_ratios=[1.5, 1, 1])

    start = float(raw_mv.annotations.onset[1]) - 2
    seg = raw_mv.copy().pick(["C3"]).crop(start, start + 30)
    axes[0].plot(seg.times + start, seg.get_data()[0] * 1e6, lw=0.6, color="#333")
    _shade_cues(axes[0], raw_mv, start, start + 30)
    for on, de in zip(raw_mv.annotations.onset, raw_mv.annotations.description):
        if de in ("T1", "T2") and start <= on <= start + 30:
            axes[0].axvline(on + config.CROP[0], color="k", lw=0.8, ls="--")
            axes[0].axvline(on + config.CROP[1], color="k", lw=0.8, ls="--")
    axes[0].set(xlabel="time (s)", ylabel="C3 (µV)",
                title=f"Cue-locked windows [{config.CROP[0]}, {config.CROP[1]}] s (dashed)")

    d = epochs_full.get_data(picks=["C3"], copy=True)[:, 0, :] * 1e6
    labels_ = epochs_full.events[:, 2]  # 1=left, 2=right
    vmax = np.percentile(np.abs(d), 98)
    for ax, cls, code in ((axes[1], "left", 1), (axes[2], "right", 2)):
        m = d[labels_ == code]
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[epochs_full.tmin, epochs_full.tmax, 0, len(m)])
        ax.axvline(0, color="k", lw=0.8)
        for x in config.CROP:
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set(title=f"C3 epochs — {cls} ({len(m)} trials)", xlabel="time from cue (s)")
    axes[1].set_ylabel("trial")
    fig.colorbar(im, ax=axes[2], label="µV", fraction=0.05)
    save_fig(fig, "s3_epoching",
             f"**S3 — Epoching (S{subject:03d}, imagery runs).** Left: the continuous "
             "band-passed signal is cut into cue-locked windows; the analysis window "
             f"[{config.CROP[0]}, {config.CROP[1]}] s skips the cue-onset transient and stays inside the "
             "~4.1 s trial. Middle/right: every extracted trial at C3 stacked as an image "
             "(one row = one trial). Single trials are dominated by trial-to-trial "
             "variability — no visible class difference at one electrode — which is exactly "
             "why the models pool spatial structure across all 64 channels.")


def fig_s4_rejection(meta):
    """Pre-rejection p2p distribution vs threshold + per-subject drop rates."""
    import numpy as np
    import pandas as pd
    p2p = np.concatenate([m["p2p"] for m in meta]) * 1e6
    thr = config.REJECT_P2P * 1e6
    drop = (pd.DataFrame([{"subject": m["subject"],
                           "dropped": m["n_total"] - m["n_kept"], "total": m["n_total"]}
                          for m in meta])
            .groupby("subject").sum())
    drop["pct"] = 100 * drop.dropped / drop.total
    overall = 100 * drop.dropped.sum() / drop.total.sum()

    n_rep = sum(1 for m in meta if m["interp"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
    axes[0].hist(p2p, bins=80, color="#777")
    axes[0].axvline(thr, color="r", ls="--", lw=1, label=f"threshold {thr:.0f} µV")
    axes[0].set(xlabel="epoch peak-to-peak (µV), max over channels", ylabel="epochs",
                title="Amplitude distribution after electrode repair")
    axes[0].legend(frameon=False)
    axes[1].bar(range(len(drop)), drop.pct.sort_values(ascending=False), width=0.9, color="#777")
    axes[1].set(xlabel="subject (sorted)", ylabel="% epochs dropped",
                title=f"Rejection impact per subject (overall {overall:.2f}%)")
    axes[1].set_xticks([])
    save_fig(fig, "s4_rejection",
             f"**S4 — Cleaning.** Two levels, both label-free. (1) Repair: a channel that "
             f"alone would reject most of a run is an electrode fault, not a trial problem — "
             f"it gets interpolated from neighbors ({n_rep} run files needed this). (2) Reject: "
             f"epochs where any remaining channel exceeds {thr:.0f} µV peak-to-peak on the "
             f"analysis window (left panel; right: per-subject drop rates, overall "
             f"{overall:.2f}%). The threshold was revised once from artifact statistics after "
             f"the initial 100 µV — calibrated to single-channel scale — rejected 75% of "
             f"epochs; decoding results never influenced it. A with/without-rejection "
             f"sensitivity check accompanies the model results.")


def fig_s5_model_input(epochs_by_class, covs, y, tangent_xy=None):
    """ERD/ERS TFR at C3 vs C4; class-mean covariances + diff; tangent t-SNE. TODO(Phase B)."""
    raise NotImplementedError


def fig_s6_model_learned(csp, lda, info, per_subject_acc):
    """CSP pattern topomaps; readout weights at channel level; accuracy distribution. TODO(Phase B)."""
    raise NotImplementedError


def fig_s7_evaluation(ladder: "pd.DataFrame", transfer: "pd.DataFrame"):
    """The ladder-collapse chart (rungs 0→3) + transfer bars. TODO(Phase B)."""
    raise NotImplementedError
