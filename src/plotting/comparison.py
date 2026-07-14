"""
GSADF vs SV-ADF comparison plot.

Both SV-ADF windows (W1 post-ChatGPT and W2 pre-ChatGPT) are shown on
the SAME panel so the two periods can be directly compared against the
GSADF result on a common timeline.

Four panels:
  1. Price — window regions and all episode shading
  2. BSADF stat + MC CV path  (GSADF window)
  3. SV-ADF coefficient paths — W1 and W2 overlaid on the same axes
  4. Episode timeline strip   — GSADF / SV W1 / SV W2 rows
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from config import settings
from src.io.data    import load_ticker_series
from src.io.results import load_gsadf, load_svadf, load_sadf_window

# Canonical colours and labels for the two configured windows
_W_COLORS = ["#e76f51", "#2a9d8f", "#6a0572"]   # orange, teal, purple
_W_LABELS = [
    "SV-ADF W1 (post-ChatGPT)",
    "SV-ADF W2 (pre-ChatGPT)",
    "SV-ADF W3",
]
_CHATGPT = pd.Timestamp("2022-11-30")


def _resolve_windows(
    svadf_window:  tuple[str, str] | None,
    svadf_windows: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Return the list of SV-ADF windows to display.

    Priority: svadf_windows > svadf_window (legacy) > settings defaults.
    """
    if svadf_windows is not None:
        return svadf_windows
    if svadf_window is not None:
        return [svadf_window]
    return [
        (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
        (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
    ]


def plot_comparison(ticker: str,
                    svadf_window:  tuple[str, str] | None = None,
                    svadf_windows: list[tuple[str, str]] | None = None,
                    save_path: Optional[Path] = None,
                    show: bool = True) -> plt.Figure | None:
    """
    GSADF + SV-ADF comparison figure for one ticker.

    Parameters
    ----------
    svadf_windows : list of (start, end) pairs — both shown on the same panel.
                    Defaults to [W1, W2] from settings.
    svadf_window  : legacy single-window argument; wrapped into a list.
    """
    windows = _resolve_windows(svadf_window, svadf_windows)

    g_data = load_gsadf(ticker)
    if g_data is None:
        print(f"[{ticker}] no GSADF results on disk")
        return None
    g_summary = g_data["summary"]
    g_paths   = g_data["paths"]

    # Load all requested SV-ADF windows (skip missing silently)
    sv_list = []
    for win in windows:
        d = load_svadf(ticker, *win)
        if d is not None:
            sv_list.append(d)

    if not sv_list:
        print(f"[{ticker}] no SV-ADF results for any requested window")
        return None

    # Common x-axis: union of GSADF + all SV-ADF windows
    g_start = pd.Timestamp(g_summary["start_date"])
    g_end   = pd.Timestamp(g_summary["end_date"])
    xmin    = min([g_start] + [pd.Timestamp(d["summary"]["start_date"]) for d in sv_list])
    xmax    = max([g_end]   + [pd.Timestamp(d["summary"]["end_date"])   for d in sv_list])

    price   = load_ticker_series(ticker, str(xmin.date()), str(xmax.date()))
    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 0.6]})
    fig.suptitle(f"{ticker}   {name}\nSegment: {seg}",
                 fontsize=11, fontweight="bold", y=1.00)

    # ── Panel 1: Price ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.1)
    ax.axvspan(g_start, g_end, alpha=0.05, color="#1f4e79", label="GSADF window")
    # GSADF episode shading
    for ep in g_summary["episodes"]:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.18, color="#c0392b")
    # SV-ADF window regions and episode shading
    for j, (d, col, lbl) in enumerate(zip(sv_list, _W_COLORS, _W_LABELS)):
        s_sum = d["summary"]
        ax.axvspan(pd.Timestamp(s_sum["start_date"]), pd.Timestamp(s_sum["end_date"]),
                   alpha=0.07, color=col, label=lbl)
        ep = s_sum.get("episode")
        if ep:
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.18, color=col)
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":", label="ChatGPT")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax.set_ylabel("Adj. close (USD)", fontsize=8)
    ax.set_title("Price — window regions and episode shading (red=GSADF)", fontsize=9)

    # ── Panel 2: BSADF (GSADF window only) ──────────────────────────────────
    ax = axes[1]
    ax.plot(g_paths.index, g_paths["bsadf_stat"], color="#1f4e79", lw=1.0, label="BSADF")
    ax.plot(g_paths.index, g_paths["cv_path"], color="orange", lw=1.0, ls="--",
            label=f"MC {int(settings.GSADF_QUANTILE*100)}% CV")
    ax.axhline(0, color="grey", lw=0.5)
    for ep in g_summary["episodes"]:
        ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                   alpha=0.22, color="red")
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("BSADF stat", fontsize=8)
    ax.set_title(f"GSADF  (T={g_summary['T']}, MC={g_summary['mc_reps']}, "
                 f"{len(g_summary['episodes'])} episode(s))", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    # ── Panel 3: SV-ADF coefficient paths — W1 and W2 on the same axes ──────
    ax = axes[2]
    n_sv_ep = 0
    for j, (d, col, lbl) in enumerate(zip(sv_list, _W_COLORS, _W_LABELS)):
        s_sum   = d["summary"]
        s_paths = d["paths"]
        ep = s_sum.get("episode")
        ls_main = "-" if j == 0 else "--"
        ax.plot(s_paths.index, s_paths["coef_stat"], color=col, lw=1.0, ls=ls_main,
                label=f"{lbl} stat")
        ax.plot(s_paths.index, s_paths["orig_thr"], color=col, lw=0.7, ls=":",
                alpha=0.55, label=f"  orig thr (log n/10)")
        if ep:
            n_sv_ep += 1
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.18, color=col, label=f"  episode [{ep['collapse_type']}]")
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.2, ls="--")
            if ep["collapse_type"] in ("post_bridge", "bridge"):
                ax.axvline(pd.Timestamp(ep["end"]), color=col, lw=1.2, ls="-.")
            else:
                ax.axvline(pd.Timestamp(ep["end"]), color="#aaa", lw=0.9, ls=":")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SV-ADF stat", fontsize=8)
    ax.set_title(f"SV-ADF — {len(sv_list)} window(s) overlaid  "
                 f"({n_sv_ep} episode(s) total)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

    # ── Panel 4: Episode timeline strip ─────────────────────────────────────
    ax = axes[3]
    strip = [("GSADF", "#c0392b", g_summary["episodes"])] + [
        (f"SV {j+1}", _W_COLORS[j],
         [d["summary"]["episode"]] if d["summary"].get("episode") else [])
        for j, d in enumerate(sv_list)
    ]
    n_rows = len(strip)
    ax.set_ylim(-0.5, n_rows - 0.5)
    for i, (lbl, col, eps) in enumerate(strip):
        ax.axhline(i, color="#ddd", lw=10, solid_capstyle="butt")
        for ep in eps:
            if ep is None:
                continue
            s = pd.Timestamp(ep["start"])
            e = pd.Timestamp(ep["end"])
            ax.barh(i, e - s, left=s, height=0.6, color=col, alpha=0.85)
        ax.text(xmin + pd.Timedelta(days=15), i, lbl,
                va="center", ha="left", fontsize=8, fontweight="bold", color=col)
    ax.set_yticks([])
    ax.set_title("Episode timeline", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for a in axes:
        a.set_xlim(xmin, xmax)

    # Grey overlay when GSADF fails to reject globally (PSY 2015)
    if not g_summary.get("reject", True):
        for a in axes:
            a.axvspan(xmin, xmax, alpha=0.18, color="#888888", zorder=4)
        axes[0].text(
            0.5, 0.97,
            "Global GSADF H₀ not rejected — GSADF episodes not date-stamped (PSY 2015)",
            transform=axes[0].transAxes,
            ha="center", va="top", fontsize=8, style="italic", color="#555555",
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


def plot_sadf_vs_svadf(ticker: str,
                        svadf_window:  tuple[str, str] | None = None,
                        svadf_windows: list[tuple[str, str]] | None = None,
                        save_path: Optional[Path] = None,
                        show: bool = True) -> plt.Figure | None:
    """
    SADF + SV-ADF comparison figure for one ticker.

    Both methods are shown on the same windows (W1 and W2) so the statistics
    are directly comparable on the same date ranges.

    Four panels:
      1. Price — window regions and SV-ADF episode shading
      2. SADF stat paths — W1 and W2 overlaid + flat asymptotic CV line
      3. SV-ADF coefficient paths — W1 and W2 overlaid + origination thresholds
      4. Timeline strip — SADF W1 peak / SADF W2 peak / SV W1 episode / SV W2 episode

    Parameters
    ----------
    svadf_windows : list of (start, end) pairs.  Defaults to [W1, W2] from settings.
    svadf_window  : legacy single-window argument; wrapped into a list.
    """
    windows = _resolve_windows(svadf_window, svadf_windows)

    # Load windowed SADF and SV-ADF for each window (skip missing silently)
    sadf_list, sv_list = [], []
    for win in windows:
        sd = load_sadf_window(ticker, *win)
        sv = load_svadf(ticker, *win)
        if sd is not None and sv is not None:
            sadf_list.append(sd)
            sv_list.append(sv)
        elif sd is not None:
            sadf_list.append(sd)
            sv_list.append(None)
        elif sv is not None:
            sadf_list.append(None)
            sv_list.append(sv)

    # Need at least one window with data from either method
    if not any(sd is not None for sd in sadf_list) and not any(sv is not None for sv in sv_list):
        print(f"[{ticker}] no windowed SADF or SV-ADF results — re-run with --methods sadf svadf")
        return None
    if not any(sd is not None for sd in sadf_list):
        print(f"[{ticker}] no windowed SADF results — re-run with --methods sadf --force")
        return None

    # Common x-axis: union of all windows
    all_summaries = [d["summary"] for d in sadf_list if d is not None] + \
                    [d["summary"] for d in sv_list   if d is not None]
    xmin = min(pd.Timestamp(s["start_date"]) for s in all_summaries)
    xmax = max(pd.Timestamp(s["end_date"])   for s in all_summaries)

    price     = load_ticker_series(ticker, str(xmin.date()), str(xmax.date()))
    seg, name = settings.SEGMENT_OF.get(ticker, ("?", ticker))

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 0.6]})
    fig.suptitle(f"{ticker}   {name}\nSegment: {seg}",
                 fontsize=11, fontweight="bold", y=1.00)

    # ── Panel 1: Price ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values, color="#1f4e79", lw=1.1)
    for j, (sd, sv, col, lbl) in enumerate(zip(sadf_list, sv_list, _W_COLORS, _W_LABELS)):
        # Window region shading (use whichever is available)
        ref_sum = (sd or sv)
        if ref_sum is None:
            continue
        r_sum = ref_sum["summary"]
        ax.axvspan(pd.Timestamp(r_sum["start_date"]), pd.Timestamp(r_sum["end_date"]),
                   alpha=0.06, color=col, label=lbl.replace("SV-ADF ", ""))
        # SADF episodes
        for ep in (sd["summary"].get("episodes", []) if sd is not None else []):
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.15, color=col)
        # SV-ADF episode
        if sv is not None:
            sv_ep = sv["summary"].get("episode")
            if sv_ep:
                ax.axvspan(pd.Timestamp(sv_ep["start"]), pd.Timestamp(sv_ep["end"]),
                           alpha=0.22, color=col, hatch="//", linewidth=0)
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":", label="ChatGPT")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax.set_ylabel("Adj. close (USD)", fontsize=8)
    ax.set_title("Price — window regions, SADF episodes (solid) and SV-ADF episodes (hatched)",
                 fontsize=9)

    # ── Panel 2: Windowed SADF paths overlaid ────────────────────────────────
    ax = axes[1]
    cv_labelled = False
    n_sadf_ep = 0
    for j, (sd, col, lbl) in enumerate(zip(sadf_list, _W_COLORS, _W_LABELS)):
        if sd is None:
            continue
        s_sum   = sd["summary"]
        s_paths = sd["paths"]
        ls_main = "-" if j == 0 else "--"
        w_lbl   = lbl.replace("SV-ADF", "SADF")
        ax.plot(s_paths.index, s_paths["sadf_stat"],
                color=col, lw=1.0, ls=ls_main, label=f"{w_lbl}")
        for ep in s_sum.get("episodes", []):
            n_sadf_ep += 1
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.22, color=col)
        if not cv_labelled:
            ax.axhline(s_sum["cv_5pct"], color="orange", lw=1.0, ls="--",
                       label=f"5% CV  ({s_sum['cv_5pct']:.2f})")
            cv_labelled = True
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SADF stat", fontsize=8)
    ax.set_title(f"SADF — {sum(sd is not None for sd in sadf_list)} window(s) overlaid  "
                 f"({n_sadf_ep} episode(s) total)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

    # ── Panel 3: SV-ADF coefficient paths ────────────────────────────────────
    ax = axes[2]
    n_sv_ep = 0
    for j, (sv, col, lbl) in enumerate(zip(sv_list, _W_COLORS, _W_LABELS)):
        if sv is None:
            continue
        s_sum   = sv["summary"]
        s_paths = sv["paths"]
        ep = s_sum.get("episode")
        ls_main = "-" if j == 0 else "--"
        ax.plot(s_paths.index, s_paths["coef_stat"], color=col, lw=1.0, ls=ls_main,
                label=f"{lbl} stat")
        ax.plot(s_paths.index, s_paths["orig_thr"], color=col, lw=0.7, ls=":",
                alpha=0.55, label="  orig thr (log n/10)")
        if ep:
            n_sv_ep += 1
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       alpha=0.18, color=col, label=f"  episode [{ep['collapse_type']}]")
            ax.axvline(pd.Timestamp(ep["start"]), color=col, lw=1.2, ls="--")
            end_col = col if ep["collapse_type"] in ("post_bridge", "bridge") else "#aaa"
            ax.axvline(pd.Timestamp(ep["end"]), color=end_col, lw=1.2,
                       ls="-." if ep["collapse_type"] in ("post_bridge", "bridge") else ":")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(_CHATGPT, color="black", lw=0.8, ls=":")
    ax.set_ylabel("SV-ADF stat", fontsize=8)
    ax.set_title(f"SV-ADF — {sum(sv is not None for sv in sv_list)} window(s) overlaid  "
                 f"({n_sv_ep} episode(s) total)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, ncol=2)

    # ── Panel 4: Timeline strip ──────────────────────────────────────────────
    ax = axes[3]
    strip = []
    for j, (sd, sv, col, lbl) in enumerate(zip(sadf_list, sv_list, _W_COLORS, _W_LABELS)):
        w_label = f"W{j+1}"
        sadf_segs = sd["summary"].get("episodes", []) if sd is not None else []
        strip.append((f"SADF {w_label}", col, sadf_segs))
        sv_segs = ([sv["summary"]["episode"]] if sv is not None and sv["summary"].get("episode")
                   else [])
        strip.append((f"SV   {w_label}", col, sv_segs))

    n_rows = len(strip)
    ax.set_ylim(-0.5, n_rows - 0.5)
    for i, (lbl, col, eps) in enumerate(strip):
        ax.axhline(i, color="#ddd", lw=10, solid_capstyle="butt")
        for ep in eps:
            if ep is None:
                continue
            s = pd.Timestamp(ep["start"])
            e = pd.Timestamp(ep["end"])
            ax.barh(i, e - s, left=s, height=0.6, color=col, alpha=0.85)
        ax.text(xmin + pd.Timedelta(days=15), i, lbl,
                va="center", ha="left", fontsize=8, fontweight="bold", color=col)
    ax.set_yticks([])
    ax.set_title("Signal / episode timeline", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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
