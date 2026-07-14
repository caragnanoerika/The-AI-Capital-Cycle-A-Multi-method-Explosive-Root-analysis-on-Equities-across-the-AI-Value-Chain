"""
Step 9b — SV-ADF parameter-sensitivity comparison figure.

Runs two configurations of the NASDAQ dot-com validation side by side
to illustrate how parameter and window choices affect episode detection:

  Left  — Sarkar's exact configuration
              window:  1982-01-01 → 2012-12-31  (T ≈ 7 560)
              M = 360, R = 360, bridge = 720 d
              Expected: origination Apr 1995, collapse Sep 2000

  Right — Alternative (proportionally scaled) configuration
              window:  1990-01-01 → 2003-12-31  (T ≈ 3 480)
              M, R, bridge proportional to T (≈ 175, 175, 245 d)

Output: outputs/figures/validation_svadf/svadf_validation_comparison.png

Usage
-----
    python scripts/09b_svadf_validation_comparison.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from config import settings
from src.methods.svadf    import svadf as run_svadf
from src.methods.episodes import detect_sv_episode
from src.io.results       import save_svadf, load_svadf
from src.plotting.diagnostic import _find_near_miss_crossings


def _scale_params_local(case: dict, T: int):
    M      = case.get("svadf_M")      or max(settings.SV_MIN_UP,   int(0.05 * T))
    R      = case.get("svadf_R")      or max(settings.SV_MIN_DOWN,  int(0.05 * T))
    bridge = case.get("svadf_bridge") or max(settings.SV_BRIDGE_DAYS, int(0.07 * T))
    return int(M), int(R), int(bridge)


FIGURES_DIR = settings.FIGURES_DIR / "validation_svadf"
EP_COL  = "#e76f51"
ALT_COL = "#2a9d8f"


def _fetch(case: dict) -> pd.Series:
    raw = yf.download(case["ticker"], start=case["start"], end=case["end"],
                      auto_adjust=True, progress=False, threads=False)
    s = raw["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index)
    return s


def _compute(name: str, case: dict, series: pd.Series, force: bool = False):
    """Run SV-ADF and return (summary, paths_df, M, R, bridge)."""
    T = len(series)
    M, R, bridge = _scale_params_local(case, T)
    w_start, w_end = case["start"], case["end"]

    existing = load_svadf(name, w_start, w_end)
    if existing and not force:
        return existing["summary"], existing["paths"], M, R, bridge

    y   = series.to_numpy()
    res = run_svadf(y)
    ep  = detect_sv_episode(
        res["coef_stat"], res["orig_thr"], res["screen_thr"], res["coll_thr"],
        dates=series.index, M=M, R=R, bridge_days=bridge,
    )
    summary = {
        "ticker": name, "window_id": f"{w_start}_{w_end}",
        "T": int(T), "start_date": series.index[0].isoformat()[:10],
        "end_date": series.index[-1].isoformat()[:10],
        "min_window": int(res["min_window"]),
        "M": M, "R": R, "bridge_days": bridge, "episode": ep,
    }
    paths_df = pd.DataFrame({
        "coef_stat": res["coef_stat"], "orig_thr": res["orig_thr"],
        "screen_thr": res["screen_thr"], "coll_thr": res["coll_thr"],
    }, index=series.index)
    paths_df.index.name = "date"
    save_svadf(name, w_start, w_end, summary, paths_df)
    return summary, paths_df, M, R, bridge


def plot_comparison() -> None:
    case_ref = settings.VALIDATION_CASES["NASDAQ_dotcom"]
    case_alt = settings.VALIDATION_CASES["NASDAQ_dotcom_alt"]

    print("Fetching NASDAQ data …")
    s_ref = _fetch(case_ref)
    s_alt = _fetch(case_alt)

    print("Running SV-ADF (reference) …")
    sum_ref, paths_ref, M_ref, R_ref, b_ref = _compute("NASDAQ_dotcom",     case_ref, s_ref, force=True)
    print("Running SV-ADF (alternative) …")
    sum_alt, paths_alt, M_alt, R_alt, b_alt = _compute("NASDAQ_dotcom_alt", case_alt, s_alt, force=True)

    ep_ref = sum_ref.get("episode")
    ep_alt = sum_alt.get("episode")

    # ── Figure: 3 rows × 2 columns ───────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 2, figsize=(16, 9), sharex="col",
        gridspec_kw={"height_ratios": [1.3, 1.5, 0.45]},
    )
    fig.suptitle(
        "SV-ADF parameter sensitivity — NASDAQ Composite dot-com episode\n"
        "Left: Sarkar's exact parameters  |  Right: proportionally scaled alternative",
        fontsize=11, fontweight="bold",
    )

    configs = [
        (axes[:, 0], s_ref, paths_ref, sum_ref, ep_ref, M_ref, R_ref, b_ref,
         EP_COL,  "Sarkar (1982–2012)",
         f"M={M_ref}, R={R_ref}, bridge={b_ref}d"),
        (axes[:, 1], s_alt, paths_alt, sum_alt, ep_alt, M_alt, R_alt, b_alt,
         ALT_COL, "Alternative (1990–2003)",
         f"M={M_alt}, R={R_alt}, bridge={b_alt}d"),
    ]

    for col_axes, series, paths, summary, ep, M, R, bridge, col, label, param_str in configs:
        xmin = pd.Timestamp(summary["start_date"])
        xmax = pd.Timestamp(summary["end_date"])

        # ── Row 0: Price ──────────────────────────────────────────────────────
        ax = col_axes[0]
        ax.plot(series.index, series.values, color="#1f4e79", lw=0.9)
        if ep:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.22, color=col)
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.3, ls="--",
                       label=f"Origination {str(ep['start'])[:7]}")
            ax.axvline(pd.Timestamp(ep["end"]),   color=col, lw=1.1, ls="-.",
                       label=f"Collapse {str(ep['end'])[:7]}")
        ax.set_title(
            f"{label}\n{param_str}  |  T={summary['T']}",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylabel("NASDAQ (USD)", fontsize=8)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
        ax.set_xlim(xmin, xmax)

        # ── Row 1: SV-ADF stat ────────────────────────────────────────────────
        ax = col_axes[1]
        stat = paths["coef_stat"].to_numpy()
        orig = paths["orig_thr"].to_numpy()
        coll = paths["coll_thr"].to_numpy()
        nm = _find_near_miss_crossings(stat, orig, M)
        nm_done = False
        for (ns, ne) in nm:
            lbl = f"Near-miss (< {M} steps)" if not nm_done else None
            ax.axvspan(paths.index[ns], paths.index[ne],
                       alpha=0.18, color="#adb5bd", label=lbl)
            nm_done = True
        ax.plot(paths.index, stat,  color=col, lw=0.9, label="SV-ADF stat")
        ax.plot(paths.index, orig,  color="#2a9d8f", lw=0.8, ls="--",
                label="Orig. thr log(τ)/10")
        ax.plot(paths.index, coll,  color="#e9c46a", lw=0.7, ls=":", alpha=0.8,
                label="Coll. thr log(τ)/2")
        ax.axhline(0, color="grey", lw=0.4)
        if ep:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.20, color=col)
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.3, ls="--")
            ax.axvline(pd.Timestamp(ep["end"]),   color=col, lw=1.1, ls="-.")
        ax.set_ylabel("SV-ADF stat", fontsize=8)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

        # ── Row 2: Timeline strip ─────────────────────────────────────────────
        ax = col_axes[2]
        ax.set_ylim(-0.5, 0.5)
        ax.axhline(0, color="#ddd", lw=10, solid_capstyle="butt")
        if ep:
            s_ts = pd.Timestamp(ep["start"]); e_ts = pd.Timestamp(ep["end"])
            ax.barh(0, e_ts - s_ts, left=s_ts, height=0.55, color=col, alpha=0.9)
            dur = (e_ts - s_ts).days
            ax.text(s_ts + pd.Timedelta(days=10), 0,
                    f"{str(ep['start'])[:7]} → {str(ep['end'])[:7]}  ({dur//365}y {(dur%365)//30}m)",
                    va="center", ha="left", fontsize=7, color=col, fontweight="bold")
        else:
            ax.text(xmin + pd.Timedelta(days=30), 0, "No episode detected",
                    va="center", ha="left", fontsize=8, color="#888", style="italic")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.set_xlim(xmin, xmax)

    # Reference line at March 2000 on both columns
    for col_axes, _, _, summary, _, _, _, _, col, _, _ in configs:
        peak = pd.Timestamp("2000-03-01")
        xmin = pd.Timestamp(summary["start_date"])
        xmax = pd.Timestamp(summary["end_date"])
        if xmin <= peak <= xmax:
            for ax in col_axes[:2]:
                ax.axvline(peak, color="#555", lw=0.8, ls=":",
                           label="NASDAQ peak (Mar 2000)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "svadf_validation_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")

    # ── Console summary ───────────────────────────────────────────────────────
    for name, summary, M, R, b in [
        ("Sarkar (reference)",  sum_ref, M_ref, R_ref, b_ref),
        ("Alternative",         sum_alt, M_alt, R_alt, b_alt),
    ]:
        ep = summary.get("episode")
        ep_str = (f"{str(ep['start'])[:7]} → {str(ep['end'])[:7]}  [{ep['collapse_type']}]"
                  if ep else "no episode")
        print(f"  {name:<25}  T={summary['T']:5d}  M={M}  R={R}  bridge={b}d"
              f"  →  {ep_str}")


if __name__ == "__main__":
    plot_comparison()
