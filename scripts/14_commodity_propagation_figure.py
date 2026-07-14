"""
14_commodity_propagation_figure.py
Figure 6 — H6: commodity-market evidence for AI-demand propagation.

Two-panel layout, computed live from data/results/ (no hardcoded numbers —
this used to be a standalone figure with hand-typed statistics; it now reads
the same cached GSADF/SV-ADF results as every other script, so it can never
silently drift from the rest of the pipeline):

  Panel A — GSADF null-rejection rate by commodity group (strong-form test)
  Panel B — Share of GSADF episode-days falling after Q4 2023, the rough
            "AI-demand era" cutoff used throughout the thesis (conditional-
            form test; see notebooks/03_summary_graphs.ipynb Figure 6 for the
            exploratory 4-group version this finalises into 6 groups)

SV-ADF W1 (post-ChatGPT window) episode detection is annotated on Panel A's
x-axis and, for whichever group has hits, called out on Panel B.

Groups are exactly config.settings.COMMODITIES's six categories, in their
settings.py order (AI-relevant energy/metals -> Confound precious metals ->
Placebo agriculture/livestock -> Benchmark), matching hypothesis H6's
AI-relevant / confound / placebo / benchmark structure.

Usage
-----
    python scripts/14_commodity_propagation_figure.py

Output: outputs/figures/thesis/fig6_commodity_propagation_v2.png
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import settings
from src.io.results import load_gsadf, load_svadf

# ─────────────────────────────────────────────────────────────────────────────
# H6 grouping & display metadata
# Keys must match config.settings.COMMODITIES exactly; dict order is plot order.
# ─────────────────────────────────────────────────────────────────────────────
GROUP_LABELS = {
    "AI-relevant: Energy (DC power proxy)":             "AI-Relevant\nEnergy",
    "AI-relevant: Industrial metals (hardware inputs)":  "AI-Relevant\nMetals",
    "Confound: Precious metals (monetary)":              "Precious Metals\n[Confound]",
    "Placebo: Agriculture":                              "Agriculture\n[Placebo]",
    "Placebo: Livestock":                                "Livestock\n[Placebo]",
    "Benchmark: Broad commodity indices":                "Broad\nBenchmark",
}

BAR_COLORS  = {"ai": "#C0392B", "confound": "#D4850A", "placebo": "#2471A3", "benchmark": "#808B96"}
EDGE_COLORS = {"ai": "#922B21", "confound": "#A0640A", "placebo": "#1A5276", "benchmark": "#5D6D7E"}

Q4_2023  = pd.Timestamp("2023-10-01")   # "post-Q4-2023" cutoff used across the thesis
SVADF_W1 = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
DOMINANT_SHARE_THRESHOLD = 0.90   # footnote-worthy if one ticker drives >= 90% of a group's post-Q4 days


def _group_class(group_key: str) -> str:
    if group_key.startswith("AI-relevant"): return "ai"
    if group_key.startswith("Confound"):    return "confound"
    if group_key.startswith("Placebo"):     return "placebo"
    return "benchmark"


def _episode_days_pre_post(episodes: list[dict], cutoff: pd.Timestamp) -> tuple[int, int]:
    """
    Split each episode's day count at `cutoff`. Episodes fully before/after the
    cutoff count entirely toward pre/post; episodes straddling it are prorated
    by the number of days on each side (mirrors notebooks/03's ep_days_split).
    """
    pre = post = 0
    for ep in episodes:
        s, e = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])
        if e < cutoff:
            pre += ep["duration_days"]
        elif s >= cutoff:
            post += ep["duration_days"]
        else:
            pre  += (cutoff - s).days
            post += (e - cutoff).days
    return pre, post


def _gsadf_episodes(ticker: str) -> list[dict]:
    """
    Raw dated BSADF episodes for `ticker`, regardless of global-null rejection.

    Unlike scripts/12_gsadf_equity_table.py (which suppresses episodes for
    tickers that fail the global GSADF supremum test, since PSY date-stamping
    is only strictly valid conditional on rejection), this exploratory
    segment-level H6 analysis follows the convention used elsewhere in the
    thesis for segment aggregates (see the equity segment table and the JJC
    discussion): a ticker whose BSADF path crosses its point-in-time critical
    value for a sustained run still produces a real, dated episode even when
    the single global supremum statistic falls short of the global CV. JJC
    (copper) is the case in point here — it never rejects the global null but
    still contributes a real 2021 episode to the AI-relevant metals group.
    """
    r = load_gsadf(ticker)
    if r is None:
        return []
    return r["summary"]["episodes"]


def build_group_stats() -> pd.DataFrame:
    """One row per H6 commodity group, with everything the figure needs."""
    rows = []
    for group_key, members in settings.COMMODITIES.items():
        tickers = list(members.keys())

        rejects: list[bool] = []
        episodes_by_ticker: dict[str, list[dict]] = {}
        for tk in tickers:
            r = load_gsadf(tk)
            if r is None:
                continue
            rejects.append(r["summary"]["reject"])
            episodes_by_ticker[tk] = _gsadf_episodes(tk)
        gsadf_rate = 100 * np.mean(rejects) if rejects else None

        all_eps = [ep for eps in episodes_by_ticker.values() for ep in eps]
        pre_d, post_d = _episode_days_pre_post(all_eps, Q4_2023)
        total_d  = pre_d + post_d
        post_pct = 100 * post_d / total_d if total_d else None

        # Per-ticker post-Q4 day contribution, to identify a dominant driver
        # (and, symmetrically, tickers that contributed nothing) for the footnote.
        per_ticker_post: dict[str, int] = {}
        per_ticker_post_range: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for tk, eps in episodes_by_ticker.items():
            _, p = _episode_days_pre_post(eps, Q4_2023)
            if p:
                per_ticker_post[tk] = p
                post_eps = [ep for ep in eps if pd.Timestamp(ep["end"]) >= Q4_2023]
                starts = [max(pd.Timestamp(ep["start"]), Q4_2023) for ep in post_eps]
                ends   = [pd.Timestamp(ep["end"]) for ep in post_eps]
                per_ticker_post_range[tk] = (min(starts), max(ends))

        dominant, dominant_share = None, 0.0
        if per_ticker_post and post_d:
            dominant = max(per_ticker_post, key=per_ticker_post.get)
            dominant_share = per_ticker_post[dominant] / post_d

        w1_hits = []
        for tk in tickers:
            r = load_svadf(tk, *SVADF_W1)
            if r is not None and r["summary"].get("episode") is not None:
                w1_hits.append((tk, r["summary"]["episode"]))

        rows.append({
            "group_key":          group_key,
            "label":              GROUP_LABELS.get(group_key, group_key),
            "class":              _group_class(group_key),
            "n":                  len(tickers),
            "gsadf_rate":         gsadf_rate,
            "post_q4_pct":        post_pct,
            "total_episode_days": total_d,
            "dominant_ticker":    dominant,
            "dominant_share":     dominant_share,
            "dominant_range":     per_ticker_post_range.get(dominant) if dominant else None,
            "zero_tickers":       [tk for tk in episodes_by_ticker if tk not in per_ticker_post],
            "svadf_w1_hits":      w1_hits,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTNOTE TEXT (built from the same numbers the bars use — nothing hand-typed)
# ─────────────────────────────────────────────────────────────────────────────
def build_footnote(gdf: pd.DataFrame) -> str:
    day_totals = " · ".join(
        f"{row['label'].replace(chr(10), ' ')} {row['total_episode_days']} d"
        for _, row in gdf.iterrows()
    )

    driver_notes = []
    for _, row in gdf.iterrows():
        if row["n"] <= 1 or row["dominant_ticker"] is None:
            continue
        if row["dominant_share"] < DOMINANT_SHARE_THRESHOLD:
            continue
        lbl = row["label"].replace("\n", " ")
        r0, r1 = row["dominant_range"]
        note = (f"* {lbl} fraction driven primarily by {row['dominant_ticker']} "
                f"({r0:%b %Y} – {r1:%b %Y}, {row['total_episode_days']} days).")
        if row["zero_tickers"]:
            note += f" {', '.join(row['zero_tickers'])} contribute zero post-Q4-2023 episode-days."
        driver_notes.append(note)

    parts = [f"Episode-day totals by group: {day_totals}."]
    parts.extend(driver_notes)
    parts.append(
        "Panel A SV-ADF W1 row (italic): episode detected (✓) or not (–) under the "
        "Sarkar-Wells (2026) volatility-robust procedure applied over the post-ChatGPT window "
        f"({SVADF_W1[0]} – {SVADF_W1[1]})."
    )
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(gdf: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 7.0), facecolor="white",
        gridspec_kw={"wspace": 0.32}
    )
    x  = np.arange(len(gdf))
    bw = 0.55
    bar_colors  = [BAR_COLORS[c]  for c in gdf["class"]]
    edge_colors = [EDGE_COLORS[c] for c in gdf["class"]]

    def style_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C0C0C0")
        ax.spines["bottom"].set_color("#C0C0C0")
        ax.tick_params(axis="both", length=0)
        ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    style_ax(ax1)
    style_ax(ax2)

    # ── Panel A — GSADF rejection rate ──────────────────────────────────────
    rates = gdf["gsadf_rate"].tolist()
    for i, val in enumerate(rates):
        if val is None:
            ax1.bar(x[i], 3.5, width=bw, color="#E8EAED", edgecolor="#9FA6AD",
                    linewidth=0.8, hatch="////", zorder=3)
            ax1.text(x[i], 5.0, "N/A", ha="center", va="bottom",
                     fontsize=9, color="#7F8C8D", style="italic")
        else:
            ax1.bar(x[i], val, width=bw, color=bar_colors[i], edgecolor=edge_colors[i],
                    linewidth=0.8, zorder=3)
            ax1.text(x[i], val + 1.5, f"{val:.0f}%", ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color="#1C2833")

    ax1.axhline(100, color="#2471A3", linestyle=":", linewidth=1.0, alpha=0.55, zorder=2)
    ax1.text(len(gdf) - 0.55, 101.5, "placebo\nlevel", fontsize=7.0, color="#2471A3",
              ha="right", va="bottom", style="italic")

    xtick_labels = []
    for _, row in gdf.iterrows():
        sv_str = "SV-ADF W1: ✓" if row["svadf_w1_hits"] else "SV-ADF W1: –"
        xtick_labels.append(f"{row['label']}\n(N={row['n']})\n{sv_str}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, fontsize=8.0, linespacing=1.4)
    ax1.tick_params(axis="x", pad=6)
    ax1.set_ylim(0, 122)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax1.set_ylabel("Share of instruments rejecting GSADF null", fontsize=9.5, labelpad=8)
    ax1.set_title("Panel A — Strong-form test\nGSADF null rejection rate by group",
                  fontsize=10, fontweight="bold", pad=12, loc="left")

    # ── Panel B — post-Q4-2023 episode-day fraction ─────────────────────────
    for i, (_, row) in enumerate(gdf.iterrows()):
        val = row["post_q4_pct"]
        if val is None:
            ax2.bar(x[i], 3.5, width=bw, color="#E8EAED", edgecolor="#9FA6AD",
                    linewidth=0.8, hatch="////", zorder=3)
            ax2.text(x[i], 5.0, "N/A", ha="center", va="bottom",
                     fontsize=9, color="#7F8C8D", style="italic")
        else:
            starred = row["n"] > 1 and row["dominant_share"] >= DOMINANT_SHARE_THRESHOLD
            lbl = f"{val:.0f}%" + (" *" if starred else "")
            ax2.bar(x[i], val, width=bw, color=bar_colors[i], edgecolor=edge_colors[i],
                    linewidth=0.8, zorder=3)
            ax2.text(x[i], val + 1.5, lbl, ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color="#1C2833")

    # Annotate the confound (precious metals) group with its SV-ADF W1 hits,
    # if any — this is where a purely GSADF-based read would be most misleading.
    confound_rows = gdf[gdf["class"] == "confound"]
    if not confound_rows.empty and confound_rows.iloc[0]["svadf_w1_hits"]:
        crow = confound_rows.iloc[0]
        ci   = gdf.index.get_loc(crow.name)
        hit_str = "\n".join(f"{tk} ({pd.Timestamp(ep['start']):%b %Y})"
                            for tk, ep in crow["svadf_w1_hits"])
        ax2.annotate(
            f"SV-ADF W1:\n{hit_str}",
            xy=(x[ci], crow["post_q4_pct"] or 0),
            xytext=(x[ci] + 1.05, 65),
            fontsize=7.5, color="#A0640A",
            arrowprops=dict(arrowstyle="->", color="#A0640A", lw=0.9,
                            connectionstyle="arc3,rad=-0.15"),
            ha="left", va="center"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{row['label']}\n(N={row['n']})" for _, row in gdf.iterrows()],
                        fontsize=8.0, linespacing=1.4)
    ax2.tick_params(axis="x", pad=6)
    ax2.set_ylim(0, 103)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax2.set_ylabel("Episode-days falling after Q4 2023 (%)", fontsize=9.5, labelpad=8)
    ax2.set_title("Panel B — Conditional-form test\nShare of episode-days after Q4 2023",
                  fontsize=10, fontweight="bold", pad=12, loc="left")

    # ── shared legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=BAR_COLORS["ai"],        edgecolor=EDGE_COLORS["ai"],        label="AI-relevant groups"),
        mpatches.Patch(facecolor=BAR_COLORS["confound"],  edgecolor=EDGE_COLORS["confound"],  label="Confound (precious metals)"),
        mpatches.Patch(facecolor=BAR_COLORS["placebo"],   edgecolor=EDGE_COLORS["placebo"],   label="Placebo groups"),
        mpatches.Patch(facecolor=BAR_COLORS["benchmark"], edgecolor=EDGE_COLORS["benchmark"], label="Broad benchmark"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, -0.09), edgecolor="#CCCCCC", framealpha=0.95)

    fig.text(0.5, -0.16, build_footnote(gdf), ha="center", fontsize=7.3,
              color="#555555", wrap=True)

    fig.suptitle(
        "Figure 6 — H6: Commodity-market evidence for AI-demand propagation\n"
        "GSADF rejection rates and post-Q4 2023 episode timing by commodity group",
        fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.20, 1, 0.99])
    return fig


if __name__ == "__main__":
    gdf = build_group_stats()
    fig = make_figure(gdf)

    out = settings.FIGURES_DIR / "thesis" / "fig6_commodity_propagation_v2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print("\n" + gdf.drop(columns=["svadf_w1_hits", "dominant_range"]).to_string(index=False))
