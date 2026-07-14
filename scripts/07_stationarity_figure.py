"""
07_stationarity_figure.py
Visualise stationarity test results across the full ticker universe.

Reads directly from data/results/stationarity/<ticker>.json (does not depend
on scripts/06_stationarity_table.py's CSV output — the two scripts are
independent consumers of the same cached JSON results).

Figure layout
─────────────
Left  : Decision heatmap (ADF | PP | KPSS)
        red  = non-stationary signal  (ADF/PP: fail to reject; KPSS: reject)
        blue = stationary signal

Right : Normalised-distance forest plot
        For ADF / PP : (stat − cv5%) / |cv5%|   > 0 → fail to reject → I(1) signal
        For KPSS     : (stat − cv5%) /  cv5%     > 0 → reject        → I(1) signal
        Sign convention is consistent: positive = I(1) evidence for every test.
        Clipped at ±8 for readability (actual KPSS ratios reach 10-15×).

Output: outputs/figures/thesis/fig_stationarity.png
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from config import settings

STAT_DIR = settings.RES_STATIONARITY_DIR
OUT_FILE = settings.FIGURES_DIR / "thesis" / "fig_stationarity.png"
CLIP     = 8.0   # normalised-distance axis limit


# ── colour palette ────────────────────────────────────────────────────────────
C_NONSTAT = "#e63946"   # red  — I(1) evidence
C_STAT    = "#457b9d"   # blue — I(0) evidence
C_ADF     = "#e63946"
C_PP      = "#2a9d8f"
C_KPSS    = "#e76f51"

SEG_COLORS = [
    "#ffe8d6", "#ffd7ba", "#f9c784", "#e9c46a",  # upstream 1-4
    "#a8dadc", "#90e0ef", "#caf0f8",              # midstream 5-7 (approx)
    "#d8f3dc", "#b7e4c7", "#95d5b2", "#52b788",  # downstream 8-11
]


def _norm_adf_pp(stat, cv):
    """(stat - cv) / |cv|  →  positive = fail to reject = I(1) signal."""
    return (stat - cv) / abs(cv) if cv != 0 else 0.0


def _norm_kpss(stat, cv):
    """(stat - cv) / cv  →  positive = reject = I(1) signal."""
    return (stat - cv) / cv if cv != 0 else 0.0


def load_results():
    rows = []
    for tk in settings.ALL_TICKERS:
        path = STAT_DIR / f"{tk}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not d:
            continue
        seg, _ = settings.SEGMENT_OF.get(tk, ("", ""))
        rows.append({
            "ticker"      : tk,
            "segment"     : seg,
            "adf_stat"    : d["adf"]["stat"],
            "adf_cv"      : d["adf"]["cv_5pct"],
            "adf_reject"  : d["adf"]["reject_5pct"],
            "pp_stat"     : d["pp"]["stat"],
            "pp_cv"       : d["pp"]["cv_5pct"],
            "pp_reject"   : d["pp"]["reject_5pct"],
            "kpss_stat"   : d["kpss"]["stat"],
            "kpss_cv"     : d["kpss"]["cv_5pct"],
            "kpss_reject" : d["kpss"]["reject_5pct"],
        })
    return rows


def make_figure(rows):
    n = len(rows)
    fig, (ax_heat, ax_forest) = plt.subplots(
        1, 2, figsize=(16, max(8, n * 0.32)),
        gridspec_kw={"width_ratios": [1, 3]},
    )
    fig.subplots_adjust(wspace=0.08)

    tickers  = [r["ticker"]  for r in rows]
    segments = [r["segment"] for r in rows]
    y_pos    = np.arange(n)

    # ── segment background bands ──────────────────────────────────────────────
    seg_idx = 0
    prev_seg = None
    band_start = 0
    seg_bands  = []
    for i, s in enumerate(segments):
        if s != prev_seg:
            if prev_seg is not None:
                seg_bands.append((band_start, i - 1, prev_seg, seg_idx % 2))
                seg_idx += 1
            band_start = i
            prev_seg   = s
    seg_bands.append((band_start, n - 1, prev_seg, seg_idx % 2))

    for ax in (ax_heat, ax_forest):
        for (s0, s1, seg, parity) in seg_bands:
            ax.axhspan(s0 - 0.5, s1 + 0.5,
                       color="#f0f0f0" if parity == 0 else "#ffffff",
                       zorder=0)

    # ── LEFT: decision heatmap ────────────────────────────────────────────────
    # Column 0: ADF  (non-stat = fail to reject = red)
    # Column 1: PP   (non-stat = fail to reject = red)
    # Column 2: KPSS (non-stat = reject         = red)
    heat = np.zeros((n, 3))
    for i, r in enumerate(rows):
        # 1 = non-stationary signal (red), 0 = stationary (blue)
        heat[i, 0] = 0 if r["adf_reject"]  else 1
        heat[i, 1] = 0 if r["pp_reject"]   else 1
        heat[i, 2] = 1 if r["kpss_reject"] else 0

    ax_heat.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest", zorder=1)

    ax_heat.set_xticks([0, 1, 2])
    ax_heat.set_xticklabels(["ADF", "PP", "KPSS"], fontsize=9, fontweight="bold")
    ax_heat.set_yticks(y_pos)
    ax_heat.set_yticklabels(tickers, fontsize=7.5)
    ax_heat.set_title("Test decisions\n(red = I(1) signal)", fontsize=9,
                       fontweight="bold", pad=6)
    ax_heat.tick_params(left=False, bottom=False)

    # segment dividers on heatmap
    for (s0, s1, seg, _) in seg_bands:
        if s0 > 0:
            ax_heat.axhline(s0 - 0.5, color="white", lw=1.5, zorder=2)

    # ── RIGHT: normalised-distance forest plot ─────────────────────────────────
    JITTER = 0.18   # vertical jitter between ADF / PP / KPSS dots

    for i, r in enumerate(rows):
        d_adf  = np.clip(_norm_adf_pp(r["adf_stat"],  r["adf_cv"]),  -CLIP, CLIP)
        d_pp   = np.clip(_norm_adf_pp(r["pp_stat"],   r["pp_cv"]),   -CLIP, CLIP)
        d_kpss = np.clip(_norm_kpss(  r["kpss_stat"], r["kpss_cv"]), -CLIP, CLIP)

        ax_forest.scatter(d_adf,  i + JITTER,  color=C_ADF,  s=30, marker="o",
                          zorder=3, alpha=0.85)
        ax_forest.scatter(d_pp,   i,            color=C_PP,   s=30, marker="s",
                          zorder=3, alpha=0.85)
        ax_forest.scatter(d_kpss, i - JITTER,  color=C_KPSS, s=30, marker="^",
                          zorder=3, alpha=0.85)

    # threshold line
    ax_forest.axvline(0, color="black", lw=1.2, ls="--", zorder=2,
                      label="Decision boundary")

    # shade I(1) region
    ax_forest.axvspan(0, CLIP, alpha=0.04, color=C_NONSTAT, zorder=1)
    ax_forest.axvspan(-CLIP, 0, alpha=0.04, color=C_STAT,    zorder=1)

    ax_forest.set_xlim(-CLIP, CLIP)
    ax_forest.set_ylim(-0.5, n - 0.5)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([])
    ax_forest.set_xlabel(
        "Normalised distance from 5% critical value\n"
        "(positive = I(1) signal;  ADF/PP: (stat−cv)/|cv|;  KPSS: (stat−cv)/cv)",
        fontsize=8)
    ax_forest.set_title(
        "Distance from rejection threshold\n(clipped at ±8 for readability)",
        fontsize=9, fontweight="bold", pad=6)
    ax_forest.tick_params(left=False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.grid(axis="x", alpha=0.25)

    # segment dividers on forest plot
    for (s0, s1, seg, _) in seg_bands:
        if s0 > 0:
            ax_forest.axhline(s0 - 0.5, color="grey", lw=0.6, ls=":", zorder=2)
        # label on right margin
        mid = (s0 + s1) / 2
        label = seg.split(".", 1)[-1].strip() if "." in seg else seg
        label = label[:30]   # truncate long names
        ax_forest.text(CLIP + 0.15, mid, label, va="center", ha="left",
                       fontsize=6.5, color="#444")

    # legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ADF,
               markersize=8, label="ADF"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PP,
               markersize=8, label="PP"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_KPSS,
               markersize=8, label="KPSS"),
        mpatches.Patch(color=C_NONSTAT, alpha=0.25, label="I(1) region"),
        mpatches.Patch(color=C_STAT,    alpha=0.25, label="I(0) region"),
    ]
    ax_forest.legend(handles=legend_handles, fontsize=8,
                     loc="lower right", framealpha=0.85)

    plt.suptitle(
        "Stationarity diagnostics — ADF, Phillips–Perron, KPSS (5% level)\n"
        f"Sample: {settings.FIXED_START} to {settings.FIXED_END}  "
        f"| {n} securities  |  all I(1) except where marked",
        fontsize=10, fontweight="bold", y=1.005)

    return fig


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    if not rows:
        print("No stationarity results found in", STAT_DIR)
        sys.exit(1)
    fig = make_figure(rows)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}  ({len(rows)} tickers)")
