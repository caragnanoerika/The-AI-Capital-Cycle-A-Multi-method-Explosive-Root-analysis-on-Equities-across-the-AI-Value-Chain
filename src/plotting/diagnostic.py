"""
Per-method diagnostic plots. All read from disk; none compute.
Modify visualization without touching the analysis layer.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from config import settings
from src.io.data    import load_ticker_series
from src.io.results import (
    load_gsadf, load_svadf, load_sadf, load_sadf_paths, list_svadf_windows,
)


def plot_sadf_only(ticker: str, save_path: Optional[Path] = None,
                   show: bool = True) -> plt.Figure | None:
    """3-panel: price + SADF path/CV + episode timeline strip (PWY date-stamping)."""
    summary = load_sadf(ticker)
    if summary is None:
        print(f"[{ticker}] no SADF results"); return None
    paths = load_sadf_paths(ticker)
    if paths is None:
        print(f"[{ticker}] no SADF paths on disk — re-run the SADF pipeline"); return None

    price    = load_ticker_series(ticker, summary["start_date"], summary["end_date"])
    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))
    episodes  = summary.get("episodes", [])
    n_ep      = len(episodes)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1.4, 1, 0.5]})
    fig.suptitle(f"{ticker}  —  {name}\nSegment: {seg}",
                 fontsize=11, fontweight="bold", y=1.00)

    # ── Panel 1: Price with episode shading ──────────────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.1)
    for ep in episodes:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color="#e76f51")
    ax.set_ylabel("Adj. close (USD)", fontsize=8)
    ep_filt = summary.get("episode_filters", {})
    ax.set_title(
        f"SADF  (T={summary['T']}, stat={summary['statistic']:.3f}, "
        f"CV={summary['cv_5pct']:.2f}, {n_ep} episode(s), "
        f"filters: min_dur={ep_filt.get('min_duration_days','?')}d, "
        f"merge={ep_filt.get('merge_gap_days','?')}d)",
        fontsize=9,
    )

    # ── Panel 2: SADF path + flat CV + episode shading ───────────────────────
    ax = axes[1]
    ax.plot(paths.index, paths["sadf_stat"], color="#1f4e79", lw=1.0, label="SADF stat")
    ax.axhline(summary["cv_5pct"], color="orange", lw=1.0, ls="--",
               label=f"5% CV  ({summary['cv_5pct']:.2f})")
    ax.axhline(0, color="grey", lw=0.5)
    for ep in episodes:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.22, color="#e76f51")
    ax.set_ylabel("SADF stat", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    # ── Panel 3: Timeline strip ───────────────────────────────────────────────
    ax = axes[2]
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#ddd", lw=10, solid_capstyle="butt")
    for ep in episodes:
        ax.barh(0, pd.Timestamp(ep["end"]) - pd.Timestamp(ep["start"]),
                left=pd.Timestamp(ep["start"]), height=0.6,
                color="#e76f51", alpha=0.85)
    ax.text(price.index[0] + pd.Timedelta(days=15), 0, "SADF",
            va="center", ha="left", fontsize=8, fontweight="bold")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    axes[0].set_xlim(price.index[0], price.index[-1])
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_gsadf_only(ticker: str, save_path: Optional[Path] = None,
                    show: bool = True) -> plt.Figure | None:
    """3-panel: price + BSADF/CV path + episode timeline strip."""
    data = load_gsadf(ticker)
    if data is None:
        print(f"[{ticker}] no GSADF results"); return None
    summary, paths = data["summary"], data["paths"]
    price = load_ticker_series(ticker, summary["start_date"], summary["end_date"])
    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1.4, 1, 0.5]})
    fig.suptitle(f"{ticker}  —  {name}\nSegment: {seg}",
                 fontsize=11, fontweight="bold", y=1.00)

    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.1)
    for ep in summary["episodes"]:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color="red")
    ax.set_ylabel("Adj. close (USD)", fontsize=8)
    ax.set_title(f"GSADF (MC={summary['mc_reps']}, "
                 f"{len(summary['episodes'])} ep, filters: "
                 f"min_dur={summary['episode_filters']['min_duration_days']}d, "
                 f"merge={summary['episode_filters']['merge_gap_days']}d)",
                 fontsize=9)

    ax = axes[1]
    ax.plot(paths.index, paths["bsadf_stat"], color="#1f4e79", lw=1.0,
            label="BSADF")
    ax.plot(paths.index, paths["cv_path"], color="orange", lw=1.0,
            ls="--", label=f"MC {int(settings.GSADF_QUANTILE*100)}% CV")
    ax.axhline(0, color="grey", lw=0.5)
    for ep in summary["episodes"]:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.22, color="red")
    ax.set_ylabel("BSADF stat", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    ax = axes[2]
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#ddd", lw=10, solid_capstyle="butt")
    for ep in summary["episodes"]:
        ax.barh(0, pd.Timestamp(ep["end"]) - pd.Timestamp(ep["start"]),
                left=pd.Timestamp(ep["start"]), height=0.6,
                color="#c0392b", alpha=0.85)
    ax.text(price.index[0] + pd.Timedelta(days=15), 0, "GSADF",
            va="center", ha="left", fontsize=8, fontweight="bold")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── Grey overlay for fail-to-reject tickers (PSY 2015) ───────────────────
    # If the global GSADF null is not rejected, any pointwise BSADF crossings
    # are not validated. Shade the entire figure grey so the red episode
    # shadings remain visible but are visually flagged as invalid.
    if not summary.get("reject", True):
        xmin, xmax = price.index[0], price.index[-1]
        for a in axes:
            a.axvspan(xmin, xmax, alpha=0.18, color="#888888", zorder=4)
        axes[0].text(
            0.5, 0.97,
            "Global H₀ not rejected — pointwise episodes not date-stamped (PSY 2015)",
            transform=axes[0].transAxes,
            ha="center", va="top", fontsize=8, style="italic",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa",
                      alpha=0.85, zorder=5),
        )

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_gsadf_segment_panel(
    segment: str,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> "plt.Figure | None":
    """
    One figure with BSADF charts for all tickers in `segment` stacked vertically.

    Layout: N rows (one per ticker) × 1 column — BSADF stat + MC CV path only.
    All rows share a common x-axis so episode dates are directly comparable.
    Figure height = N × ROW_H (fixed); width is fixed at 13".
    Tickers with no GSADF results on disk are skipped (noted in the title).
    """
    ROW_H = 1.6  # inches per ticker — compact but readable

    tickers   = [tk for tk, (seg, _) in settings.SEGMENT_OF.items() if seg == segment]
    available = [tk for tk in tickers if load_gsadf(tk) is not None]

    if not available:
        print(f"[segment '{segment}'] no GSADF results for any ticker")
        return None

    N   = len(available)
    fig, axes = plt.subplots(
        N, 1,
        figsize=(13.0, max(N * ROW_H, 2.5)),
        sharex=True, squeeze=False,
    )

    missing = [tk for tk in tickers if tk not in available]
    title   = f"GSADF episodes — {segment}"
    if missing:
        title += f"  [no results: {', '.join(missing)}]"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    all_summaries = [load_gsadf(tk)["summary"] for tk in available]
    xmin = min(pd.Timestamp(s["start_date"]) for s in all_summaries)
    xmax = max(pd.Timestamp(s["end_date"])   for s in all_summaries)

    for i, tk in enumerate(available):
        data    = load_gsadf(tk)
        summary = data["summary"]
        paths   = data["paths"]
        _, name = settings.SEGMENT_OF.get(tk, ("?", tk))
        short   = (name[:40] + "…") if len(name) > 40 else name

        ax = axes[i, 0]

        p1, = ax.plot(paths.index, paths["bsadf_stat"],
                      color="#1f4e79", lw=0.9, label="BSADF")
        p2, = ax.plot(paths.index, paths["cv_path"],
                      color="orange",  lw=0.9, ls="--",
                      label=f"CV {int(settings.GSADF_QUANTILE * 100)}%")
        ax.axhline(0, color="grey", lw=0.4)

        for ep in summary["episodes"]:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.25, color="red")

        n_ep   = len(summary["episodes"])
        ep_str = f"{n_ep} ep." if n_ep else "no ep."
        ep_col = "#c0392b" if n_ep else "#888888"

        ax.set_title(
            f"{tk} — {short}",
            loc="left", fontsize=8, fontweight="bold", pad=2,
        )
        ax.text(0.995, 0.88, ep_str, transform=ax.transAxes,
                fontsize=7, color=ep_col, ha="right", va="top",
                fontweight="bold")
        ax.tick_params(labelsize=7)

        # Grey overlay for fail-to-reject rows (PSY 2015)
        if not summary.get("reject", True):
            ax.axvspan(xmin, xmax, alpha=0.18, color="#888888", zorder=4)
            ax.text(0.5, 0.97, "H₀ not rejected",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.5, style="italic", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#aaaaaa", alpha=0.85, zorder=5))

        if i == 0:
            ax.legend(handles=[p1, p2], fontsize=7, framealpha=0.7,
                      loc="upper right", ncol=2,
                      bbox_to_anchor=(0.995, 0.55))

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].set_xlim(xmin, xmax)  # propagates to all rows via sharex

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.4)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_svadf_only(ticker: str,
                    window:  tuple[str, str] | None = None,
                    windows: list[tuple[str, str]] | None = None,
                    save_path: Optional[Path] = None,
                    show: bool = True) -> plt.Figure | None:
    """2-panel: price + SV-ADF coefficient paths.

    Both W1 and W2 windows are overlaid on the same panel by default.

    Parameters
    ----------
    windows : list of (start, end) to overlay — defaults to [W1, W2] from settings.
    window  : legacy single-window argument; wrapped into a list.
    """
    # Resolve windows
    if windows is None:
        if window is not None:
            windows = [window]
        else:
            windows = [
                (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
                (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
            ]

    data_list = [load_svadf(ticker, *w) for w in windows]
    data_list = [d for d in data_list if d is not None]

    if not data_list:
        avail = list_svadf_windows(ticker)
        if not avail:
            print(f"[{ticker}] no SV-ADF results"); return None
        data_list = [load_svadf(ticker, *w) for w in avail]
        data_list = [d for d in data_list if d is not None]

    if not data_list:
        return None

    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))
    xmin = min(pd.Timestamp(d["summary"]["start_date"]) for d in data_list)
    xmax = max(pd.Timestamp(d["summary"]["end_date"])   for d in data_list)
    price = load_ticker_series(ticker, str(xmin.date()), str(xmax.date()))

    W_COLORS = ["#e76f51", "#2a9d8f", "#6a0572"]
    W_LABELS = ["SV-ADF W1 (post-ChatGPT)", "SV-ADF W2 (pre-ChatGPT)", "SV-ADF W3"]
    CHATGPT  = pd.Timestamp("2022-11-30")

    win_str = "  |  ".join(
        f"{d['summary']['start_date']}→{d['summary']['end_date']}" for d in data_list
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.3]})
    fig.suptitle(f"{ticker}  —  {name}\nSegment: {seg}  |  SV-ADF: {win_str}",
                 fontsize=9, fontweight="bold", y=1.01)

    # ── Panel 1: Price with episode origination lines ────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.1)
    for j, (d, col, lbl) in enumerate(zip(data_list, W_COLORS, W_LABELS)):
        ep = d["summary"].get("episode")
        if ep:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.16, color=col)
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.3, ls="--",
                       label=f"{lbl} origination")
            end_col = col if ep["collapse_type"] in ("post_bridge", "bridge") else "#aaa"
            ax.axvline(pd.Timestamp(ep["end"]), color=end_col, lw=1.0,
                       ls="-." if ep["collapse_type"] in ("post_bridge", "bridge") else ":")
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":", label="ChatGPT launch")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_ylabel("Adj. close (USD)", fontsize=8)

    # ── Panel 2: SV-ADF coefficient paths — all windows on the same axes ────
    ax = axes[1]
    for j, (d, col, lbl) in enumerate(zip(data_list, W_COLORS, W_LABELS)):
        s_sum   = d["summary"]
        s_paths = d["paths"]
        ep = s_sum.get("episode")
        ls_main = "-" if j == 0 else "--"
        ax.plot(s_paths.index, s_paths["coef_stat"], color=col, lw=1.1,
                ls=ls_main, label=f"{lbl}")
        ax.plot(s_paths.index, s_paths["orig_thr"], color=col, lw=0.7,
                ls=":", alpha=0.55, label="  orig thr log(n)/10")
        ax.plot(s_paths.index, s_paths["coll_thr"], color=col, lw=0.7,
                ls=":", alpha=0.3)
        if ep:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.18, color=col, label=f"  episode [{ep['collapse_type']}]")
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.3, ls="--")
            end_col = col if ep["collapse_type"] in ("post_bridge", "bridge") else "#aaa"
            ax.axvline(pd.Timestamp(ep["end"]), color=end_col, lw=1.0,
                       ls="-." if ep["collapse_type"] in ("post_bridge", "bridge") else ":")
    ax.axhline(0, color="grey", lw=0.5)
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SV-ADF stat", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[0].set_xlim(xmin, xmax)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SV-ADF segment panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_svadf_segment_panel(
    segment: str,
    windows: list[tuple[str, str]] | None = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> "plt.Figure | None":
    """
    One figure with SV-ADF coefficient-stat charts for every ticker in `segment`.

    Each row shows the coef_stat path(s) and the orig_thr for the configured
    windows, with episode shading.  Rows share a common x-axis.

    Parameters
    ----------
    windows : list of (start, end) windows to overlay.
              Defaults to [W1, W2] from settings.
    """
    if windows is None:
        windows = [
            (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
            (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
        ]

    W_COLORS = ["#e76f51", "#2a9d8f", "#6a0572"]
    W_LABELS = ["W1 post-ChatGPT", "W2 pre-ChatGPT", "W3"]
    CHATGPT  = pd.Timestamp("2022-11-30")

    ROW_H = 1.8

    tickers   = [tk for tk, (seg, _) in settings.SEGMENT_OF.items() if seg == segment]
    # keep only tickers that have at least one window on disk
    available = [
        tk for tk in tickers
        if any(load_svadf(tk, *w) is not None for w in windows)
    ]

    if not available:
        print(f"[segment '{segment}'] no SV-ADF results for any ticker")
        return None

    N   = len(available)
    fig, axes = plt.subplots(
        N, 1,
        figsize=(13.0, max(N * ROW_H, 3.0)),
        sharex=True, squeeze=False,
    )

    missing = [tk for tk in tickers if tk not in available]
    title = f"SV-ADF — {segment}"
    if missing:
        title += f"  [no results: {', '.join(missing)}]"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Global x-range: union over all tickers and windows
    all_starts, all_ends = [], []
    for tk in available:
        for w in windows:
            d = load_svadf(tk, *w)
            if d is not None:
                all_starts.append(pd.Timestamp(d["summary"]["start_date"]))
                all_ends.append(pd.Timestamp(d["summary"]["end_date"]))
    xmin = min(all_starts)
    xmax = max(all_ends)

    for i, tk in enumerate(available):
        _, name = settings.SEGMENT_OF.get(tk, ("?", tk))
        short   = (name[:40] + "…") if len(name) > 40 else name
        ax      = axes[i, 0]

        legend_handles = []
        n_ep = 0
        for j, (w, col, lbl) in enumerate(zip(windows, W_COLORS, W_LABELS)):
            d = load_svadf(tk, *w)
            if d is None:
                continue
            s_paths = d["paths"]
            ep      = d["summary"].get("episode")
            ls      = "-" if j == 0 else "--"
            p, = ax.plot(s_paths.index, s_paths["coef_stat"],
                         color=col, lw=0.9, ls=ls, label=lbl)
            ax.plot(s_paths.index, s_paths["orig_thr"],
                    color=col, lw=0.6, ls=":", alpha=0.5)
            legend_handles.append(p)
            if ep:
                n_ep += 1
                ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                           alpha=0.20, color=col)
                ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.0, ls="--")

        ax.axhline(0, color="grey", lw=0.4)
        if xmin <= CHATGPT <= xmax:
            ax.axvline(CHATGPT, color="black", lw=0.6, ls=":")

        ep_str = f"{n_ep} ep." if n_ep else "no ep."
        ep_col = "#c0392b" if n_ep else "#888"
        ax.set_title(f"{tk} — {short}", loc="left", fontsize=8, fontweight="bold", pad=2)
        ax.text(0.995, 0.88, ep_str, transform=ax.transAxes,
                fontsize=7, color=ep_col, ha="right", va="top", fontweight="bold")
        ax.tick_params(labelsize=7)
        if i == 0 and legend_handles:
            ax.legend(handles=legend_handles, fontsize=7, framealpha=0.7,
                      loc="upper right", ncol=len(legend_handles),
                      bbox_to_anchor=(0.995, 0.55))

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].set_xlim(xmin, xmax)

    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.4)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker SV-ADF + volatility
# ─────────────────────────────────────────────────────────────────────────────

def _find_near_miss_crossings(
    stat: "np.ndarray", thr: "np.ndarray", M: int
) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs where stat > thr for a contiguous run
    of length 1 … M-1.  These are crossings too short to trigger origination
    — i.e. volatility spikes that SV-ADF correctly ignores.
    """
    above = np.isfinite(stat) & (stat > thr)
    runs  = []
    in_run, s = False, 0
    for k, a in enumerate(above):
        if a and not in_run:
            s = k; in_run = True
        elif not a and in_run:
            length = k - s
            if 1 <= length < M:
                runs.append((s, k - 1))
            in_run = False
    if in_run:
        length = len(above) - s
        if 1 <= length < M:
            runs.append((s, len(above) - 1))
    return runs


def plot_svadf_volatility(
    ticker: str,
    window:      tuple[str, str] | None = None,
    vol_window:  int = 40,
    price_series: "pd.Series | None" = None,
    save_path:   Optional[Path] = None,
    show: bool = True,
) -> "plt.Figure | None":
    """
    Four-panel figure: price / rolling volatility / SV-ADF stat / timeline.

    The volatility panel shows rolling annualised return std.  The SV-ADF
    panel annotates 'near-miss' threshold crossings (runs shorter than M)
    in grey — these are the high-volatility spikes the method avoids
    classifying as bubbles.

    Parameters
    ----------
    window       : (start, end) SV-ADF window to display. Defaults to W1.
    vol_window   : trading-day window for rolling vol (default 40 ≈ 8 weeks).
    price_series : optional pre-loaded price Series.  When supplied, bypasses
                   the load_ticker_series disk lookup — useful for validation
                   cases whose data is not in the main prices.csv.
    """
    if window is None:
        window = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)

    d = load_svadf(ticker, *window)
    if d is None:
        avail = list_svadf_windows(ticker)
        if not avail:
            print(f"[{ticker}] no SV-ADF results"); return None
        d = load_svadf(ticker, *avail[0])

    summary = d["summary"]
    paths   = d["paths"]
    ep      = summary.get("episode")

    xmin  = pd.Timestamp(summary["start_date"])
    xmax  = pd.Timestamp(summary["end_date"])
    if price_series is not None:
        price = price_series.loc[str(xmin.date()):str(xmax.date())].dropna()
    else:
        price = load_ticker_series(ticker, str(xmin.date()), str(xmax.date()))

    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))
    CHATGPT   = pd.Timestamp("2022-11-30")
    EP_COL    = "#e76f51"
    VOL_COL   = "#6c757d"

    # Rolling annualised vol (daily log returns × sqrt(252) × 100)
    log_ret    = np.log(price / price.shift(1)).dropna()
    rolling_vol = log_ret.rolling(vol_window).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.reindex(price.index)

    # Near-miss crossings
    stat_arr  = paths["coef_stat"].to_numpy()
    orig_arr  = paths["orig_thr"].to_numpy()
    M         = settings.SV_MIN_UP
    near_miss = _find_near_miss_crossings(stat_arr, orig_arr, M)

    # High-vol regimes: rolling_vol above 75th percentile → shade as "regime"
    vol_vals   = rolling_vol.dropna()
    vol_thr75  = vol_vals.quantile(0.75)
    vol_series = rolling_vol.reindex(paths.index)

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 0.8, 1.2, 0.5]},
    )
    fig.suptitle(
        f"{ticker}  —  {name}\nSV-ADF window: {summary['start_date']} → "
        f"{summary['end_date']}  |  M={M}  |  vol window={vol_window}d",
        fontsize=10, fontweight="bold", y=1.01,
    )

    # ── Panel 1: Price ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.0)
    if ep:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color=EP_COL, label="Detected episode")
        ax.axvline(pd.Timestamp(ep["start"]), color=EP_COL, lw=1.3, ls="--")
        ax.axvline(pd.Timestamp(ep["end"]),   color=EP_COL, lw=1.0, ls="-.")
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":", label="ChatGPT launch")
    ax.set_ylabel("Price (USD)", fontsize=8)
    ax.set_title("Adjusted close price", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    # ── Panel 2: Rolling volatility ──────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(price.index, rolling_vol.values,
                    alpha=0.35, color=VOL_COL, label=f"{vol_window}d rolling vol")
    ax.plot(price.index, rolling_vol.values, color=VOL_COL, lw=0.8)
    ax.axhline(vol_thr75, color=VOL_COL, lw=0.8, ls="--", alpha=0.7,
               label=f"75th pct ({vol_thr75:.0f}%)")
    # shade high-vol regimes
    in_hv = False; hv_s = None
    for dt, v in zip(price.index, rolling_vol.values):
        if not np.isnan(v) and v > vol_thr75 and not in_hv:
            hv_s = dt; in_hv = True
        elif (np.isnan(v) or v <= vol_thr75) and in_hv:
            ax.axvspan(hv_s, dt, alpha=0.12, color=VOL_COL)
            in_hv = False
    if in_hv:
        ax.axvspan(hv_s, price.index[-1], alpha=0.12, color=VOL_COL)
    if ep:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.15, color=EP_COL)
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("Ann. vol (%)", fontsize=8)
    ax.set_title(f"Rolling {vol_window}-day annualised volatility", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    # ── Panel 3: SV-ADF stat + thresholds ───────────────────────────────────
    ax = axes[2]
    # Near-miss crossings: grey shading
    nm_labeled = False
    for (ns, ne) in near_miss:
        lbl = "Near-miss (vol spike, no bubble)" if not nm_labeled else None
        ax.axvspan(paths.index[ns], paths.index[ne],
                   alpha=0.18, color="#adb5bd", label=lbl)
        nm_labeled = True
    ax.plot(paths.index, paths["coef_stat"],
            color=EP_COL, lw=1.0, label="SV-ADF coef stat")
    ax.plot(paths.index, paths["orig_thr"],
            color="#2a9d8f", lw=0.8, ls="--", label="Orig. thr log(τ)/10")
    ax.plot(paths.index, paths["screen_thr"],
            color="#f4a261", lw=0.7, ls=":", alpha=0.7, label="Screen thr log(τ)")
    ax.plot(paths.index, paths["coll_thr"],
            color="#e9c46a", lw=0.7, ls=":", alpha=0.6, label="Coll. thr log(τ)/2")
    ax.axhline(0, color="grey", lw=0.4)
    if ep:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.20, color=EP_COL, label=f"Episode [{ep['collapse_type']}]")
        ax.axvline(pd.Timestamp(ep["start"]), color=EP_COL, lw=1.3, ls="--")
        ax.axvline(pd.Timestamp(ep["end"]),   color=EP_COL, lw=1.0, ls="-.")
    if xmin <= CHATGPT <= xmax:
        ax.axvline(CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SV-ADF stat", fontsize=8)
    ax.set_title(
        f"Recursive SV-ADF  |  grey = near-miss crossing (<{M} consecutive steps)",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

    # ── Panel 4: Timeline strip ──────────────────────────────────────────────
    ax = axes[3]
    rows = [
        ("Episode", EP_COL, [ep] if ep else []),
        ("Near-miss", "#adb5bd",
         [{"start": paths.index[s], "end": paths.index[e]} for s, e in near_miss]),
        ("High vol", VOL_COL, []),   # filled below
    ]
    # Build high-vol segments for timeline
    hv_segs = []
    in_hv = False; hv_s = None
    for dt, v in zip(price.index, rolling_vol.values):
        if not np.isnan(v) and v > vol_thr75 and not in_hv:
            hv_s = dt; in_hv = True
        elif (np.isnan(v) or v <= vol_thr75) and in_hv:
            hv_segs.append({"start": hv_s, "end": dt}); in_hv = False
    if in_hv:
        hv_segs.append({"start": hv_s, "end": price.index[-1]})
    rows[2] = ("High vol", VOL_COL, hv_segs)

    n_rows = len(rows)
    ax.set_ylim(-0.5, n_rows - 0.5)
    for ri, (lbl, col, segs) in enumerate(rows):
        ax.axhline(ri, color="#ddd", lw=10, solid_capstyle="butt")
        for seg_ep in segs:
            if seg_ep is None:
                continue
            s = pd.Timestamp(seg_ep["start"])
            e = pd.Timestamp(seg_ep["end"])
            ax.barh(ri, e - s, left=s, height=0.55, color=col, alpha=0.85)
        ax.text(xmin + pd.Timedelta(days=10), ri, lbl,
                va="center", ha="left", fontsize=7, fontweight="bold", color=col)
    ax.set_yticks([])
    ax.set_title("Timeline", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for a in axes:
        a.set_xlim(xmin, xmax)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
