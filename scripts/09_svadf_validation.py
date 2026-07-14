"""
Step 9 — SV-ADF standalone validation on historical bubble episodes.

Runs the SV-ADF method on each VALIDATION_CASE using parameters scaled
to the length of the historical series (longer series need larger M / R /
bridge_days to match Sarkar's calibration rationale).

For the NASDAQ dot-com case (1990-2003, ~3 500 trading days) Sarkar's
reference code uses M=360, R=360, bridge=720 for his 1982-2012 run
(~7 500 days).  We scale proportionally:
    M         ≈ 5% of T
    R         ≈ 5% of T
    bridge    ≈ 7% of T calendar days (capped at reasonable values)

A standalone four-panel figure is saved for each case:
    1. Price + episode shading
    2. Rolling volatility
    3. SV-ADF stat with all three thresholds + near-miss annotations
    4. Timeline strip

The paper's expected dates (from settings.VALIDATION_CASES) are printed
to console for direct comparison.

Usage
-----
    python scripts/09_svadf_validation.py
    python scripts/09_svadf_validation.py --case NASDAQ_dotcom
    python scripts/09_svadf_validation.py --M 360 --R 180 --bridge 720
"""
from __future__ import annotations
import sys
import argparse
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import settings
from src.methods.svadf    import svadf as run_svadf
from src.methods.episodes import detect_sv_episode
from src.plotting.diagnostic import (
    _find_near_miss_crossings, plot_svadf_volatility,
)
from src.io.results import save_svadf, load_svadf

FIGURES_DIR = settings.FIGURES_DIR / "validation_svadf"


def fetch_series(case: dict) -> pd.Series | None:
    raw = yf.download(case["ticker"], start=case["start"], end=case["end"],
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return None
    s = raw["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index)
    s.name = case["ticker"]
    return s


def _scale_params(case: dict, T: int, M_override, R_override, bridge_override):
    """
    Return (M, R, bridge_days).

    Priority: CLI override > per-case value in settings > proportional scaling.
    For the NASDAQ dot-com case Sarkar uses M=360, R=360, bridge=720 on a
    ~7 500-day series; those values are stored as svadf_M/R/bridge in the case
    dict and will be used automatically when no CLI override is supplied.
    """
    M      = M_override      or case.get("svadf_M")      or max(settings.SV_MIN_UP,   int(0.05 * T))
    R      = R_override      or case.get("svadf_R")      or max(settings.SV_MIN_DOWN,  int(0.05 * T))
    bridge = bridge_override or case.get("svadf_bridge") or max(settings.SV_BRIDGE_DAYS, int(0.07 * T))
    return int(M), int(R), int(bridge)


def run_and_plot_case(name: str, case: dict,
                      M_override=None, R_override=None, bridge_override=None,
                      force: bool = True) -> None:
    print(f"\n{'='*70}")
    print(f"  {name}  |  {case['ticker']}  ({case['start']} → {case['end']})")
    print(f"  Expected: {case.get('expected_dates', '—')}")
    print(f"{'='*70}")

    series = fetch_series(case)
    if series is None:
        print("  [WARN] download failed"); return

    T = len(series)
    M, R, bridge = _scale_params(case, T, M_override, R_override, bridge_override)
    print(f"  T={T}  →  M={M}  R={R}  bridge={bridge}d")

    # Persist series to prices.csv so load_ticker_series can find it
    if settings.PRICES_FILE.exists():
        prices_df = pd.read_csv(settings.PRICES_FILE, index_col=0, parse_dates=True)
    else:
        prices_df = pd.DataFrame()
    prices_df[name] = series
    settings.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(settings.PRICES_FILE)

    # Run SV-ADF
    w_start, w_end = case["start"], case["end"]
    existing = load_svadf(name, w_start, w_end)
    if existing is not None and not force:
        print("  cached — skipping computation")
        summary = existing["summary"]
        paths_df = existing["paths"]
    else:
        y   = series.to_numpy()
        res = run_svadf(y)
        if res is None:
            print("  [WARN] svadf returned None"); return

        ep = detect_sv_episode(
            res["coef_stat"], res["orig_thr"], res["screen_thr"], res["coll_thr"],
            dates=series.index, M=M, R=R, bridge_days=bridge,
        )
        summary = {
            "ticker":      name,
            "window_id":   f"{w_start}_{w_end}",
            "T":           int(T),
            "start_date":  series.index[0].isoformat()[:10],
            "end_date":    series.index[-1].isoformat()[:10],
            "min_window":  int(res["min_window"]),
            "M":           M, "R": R, "bridge_days": bridge,
            "episode":     ep,
        }
        paths_df = pd.DataFrame({
            "coef_stat":  res["coef_stat"],
            "orig_thr":   res["orig_thr"],
            "screen_thr": res["screen_thr"],
            "coll_thr":   res["coll_thr"],
        }, index=series.index)
        paths_df.index.name = "date"
        save_svadf(name, w_start, w_end, summary, paths_df)

    ep = summary.get("episode")
    if ep:
        print(f"  Episode detected: {ep['start']} → {ep['end']}  [{ep['collapse_type']}]")
    else:
        print("  No episode detected")

    # ── Volatility + SV-ADF validation figure ────────────────────────────────
    fig_path = FIGURES_DIR / f"svadf_validation_{name}.png"
    fig = plot_svadf_volatility(
        ticker=name,
        window=(w_start, w_end),
        price_series=series,
        save_path=fig_path,
        show=False,
    )
    if fig is not None:
        print(f"  saved → {fig_path}")

    # ── Standalone 3-panel figure (replicating Sarkar Fig 9 style) ───────────
    _plot_standalone(name, case, summary, paths_df, series, M, ep)


def _plot_standalone(name, case, summary, paths_df, series, M, ep) -> None:
    """Price + SV-ADF stat + timeline — clean academic style."""
    CHATGPT = pd.Timestamp("2022-11-30")
    EP_COL  = "#e76f51"

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.3, 0.5]},
    )
    fig.suptitle(
        f"SV-ADF Validation — {name}  ({case['start']} → {case['end']})\n"
        f"M={summary['M']}  R={summary['R']}  bridge={summary['bridge_days']}d  "
        f"|  Expected: {case.get('expected_dates', '—')}",
        fontsize=9, fontweight="bold", y=1.01,
    )

    xmin = pd.Timestamp(summary["start_date"])
    xmax = pd.Timestamp(summary["end_date"])

    # Panel 1: Price
    ax = axes[0]
    ax.plot(series.index, series.values, color="#1f4e79", lw=1.0)
    if ep:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color=EP_COL, label=f"Detected origination → collapse")
        ax.axvline(pd.Timestamp(ep["start"]), color=EP_COL, lw=1.3, ls="--",
                   label=f"Origination {ep['start'][:7]}")
        ax.axvline(pd.Timestamp(ep["end"]),   color=EP_COL, lw=1.0, ls="-.")
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":", label="ChatGPT")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_ylabel("Price (USD)", fontsize=8)
    ax.set_title("Adjusted close price", fontsize=9)

    # Panel 2: SV-ADF stat
    ax = axes[1]
    near_miss = _find_near_miss_crossings(
        paths_df["coef_stat"].to_numpy(),
        paths_df["orig_thr"].to_numpy(),
        M,
    )
    nm_labeled = False
    for (ns, ne) in near_miss:
        lbl = "Near-miss (< M consecutive)" if not nm_labeled else None
        ax.axvspan(paths_df.index[ns], paths_df.index[ne],
                   alpha=0.15, color="#adb5bd", label=lbl)
        nm_labeled = True
    ax.plot(paths_df.index, paths_df["coef_stat"],
            color=EP_COL, lw=1.0, label="SV-ADF coef stat")
    ax.plot(paths_df.index, paths_df["orig_thr"],
            color="#2a9d8f", lw=0.9, ls="--", label="Orig. thr log(τ)/10")
    ax.plot(paths_df.index, paths_df["coll_thr"],
            color="#e9c46a", lw=0.8, ls=":", alpha=0.8, label="Coll. thr log(τ)/2")
    ax.axhline(0, color="grey", lw=0.4)
    if ep:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color=EP_COL)
        ax.axvline(pd.Timestamp(ep["start"]), color=EP_COL, lw=1.3, ls="--")
        ax.axvline(pd.Timestamp(ep["end"]),   color=EP_COL, lw=1.0, ls="-.")
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SV-ADF stat", fontsize=8)
    ax.set_title(
        f"Recursive SV-ADF  (T={summary['T']})  |  "
        f"grey = near-miss crossing (< M={M} steps)",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

    # Panel 3: Timeline
    ax = axes[2]
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#ddd", lw=10, solid_capstyle="butt")
    if ep:
        s = pd.Timestamp(ep["start"]); e = pd.Timestamp(ep["end"])
        ax.barh(0, e - s, left=s, height=0.55, color=EP_COL, alpha=0.9)
    ax.text(xmin + pd.Timedelta(days=30), 0, "SV-ADF",
            va="center", ha="left", fontsize=8, fontweight="bold", color=EP_COL)
    ax.set_yticks([])
    ax.set_title("Episode timeline", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for a in axes:
        a.set_xlim(xmin, xmax)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"svadf_validation_{name}_standalone.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved standalone → {fig_path}")


def main(cases=None, M=None, R=None, bridge=None, force=True):
    cases = cases or list(settings.VALIDATION_CASES.keys())
    for name in cases:
        case = settings.VALIDATION_CASES.get(name)
        if case is None:
            print(f"  [WARN] unknown case '{name}'"); continue
        run_and_plot_case(name, case,
                          M_override=M, R_override=R, bridge_override=bridge,
                          force=force)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case",   nargs="*", default=None,
                   help="Validation case name(s) (default: all)")
    p.add_argument("--M",      type=int, default=None,
                   help="Override origination run length M")
    p.add_argument("--R",      type=int, default=None,
                   help="Override post-bridge run length R")
    p.add_argument("--bridge", type=int, default=None,
                   help="Override bridge period in calendar days")
    p.add_argument("--no-force", dest="force", action="store_false",
                   help="Skip if cached result already exists")
    args = p.parse_args()
    main(cases=args.case, M=args.M, R=args.R, bridge=args.bridge, force=args.force)
