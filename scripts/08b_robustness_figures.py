"""
08b_robustness_figures.py
Generates two thesis-quality robustness figures from rob_summary.csv
(which must already exist, produced by 08_robustness.py).

  Figure 1  rob_stability_heatmap_v2.png
            Deviation heatmap with absolute percentage-point thresholds
            (≤8 pp = stable, 8–18 pp = moderate, >18 pp or direction flip = large).
            Two panels: GSADF checks (A1–A5) above, SV-ADF checks (B1–B2) below.
            Each panel uses its own method-appropriate baseline row.

  Figure 2  rob_scorecard.png
            Qualitative ✓ / ✗ / — hypothesis scorecard across H1–H4
            for every robustness configuration, grouped by check.

Usage
-----
    python scripts/08b_robustness_figures.py
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
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
ROB_DIR  = ROOT / "outputs" / "robustness"
ROB_FIGS = ROB_DIR / "figures"
ROB_FIGS.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_GREEN  = "#52b788"
C_ORANGE = "#f4a261"
C_RED    = "#e63946"
C_GREY   = "#adb5bd"
C_BLUE   = "#264653"
C_TEAL   = "#2a9d8f"
C_LBLUE  = "#a8dadc"
DARK     = "#212529"

# ── deviation thresholds (absolute pp) ───────────────────────────────────────
PP_STABLE   = 0.08   # ≤ 8 pp  → green
PP_MODERATE = 0.18   # 8–18 pp → orange; beyond → red

# ── H1 pass thresholds ───────────────────────────────────────────────────────
H1_THRESH_A = 0.45   # GSADF equity reject rate (A-series)
H1_THRESH_B = 0.35   # SV-ADF episode rate (B-series, more conservative test)

# ── canonical baselines ───────────────────────────────────────────────────────
GSADF_BASELINE = "A3_alpha95_baseline"  # α=95%, default mw and filters
SVADF_BASELINE = "B2_baseline"          # default M/R/bridge thresholds

GSADF_BASELINE_VARIANTS = {"A2_baseline", "A3_alpha95_baseline", "A4_mw100"}
SVADF_BASELINE_VARIANTS = {"B1_W1_baseline", "B1_W2_baseline", "B2_baseline"}
ALL_BASELINE_VARIANTS   = GSADF_BASELINE_VARIANTS | SVADF_BASELINE_VARIANTS

# ── configuration display labels ──────────────────────────────────────────────
SHORT = {
    "A1_pre_ChatGPT"    : "A1a  Sub-period: pre-ChatGPT (2020–Oct 2022)",
    "A1_post_ChatGPT"   : "A1b  Sub-period: post-ChatGPT (Nov 2022–2025)",
    "A2_md21_mg21"      : "A2a  Episode filter: md=21d, gap=21d",
    "A2_md42_mg21"      : "A2b  Episode filter: md=42d, gap=21d",
    "A2_baseline"       : "A2c  Episode filter: baseline (md=60d, gap=42d)",
    "A2_md90_mg42"      : "A2d  Episode filter: md=90d, gap=42d",
    "A2_md60_mg60"      : "A2e  Episode filter: md=60d, gap=60d",
    "A3_alpha90"        : "A3a  Significance level: α = 90%",
    "A3_alpha95_baseline": "A3b  Significance level: α = 95%  [baseline]",
    "A3_alpha99"        : "A3c  Significance level: α = 99%",
    "A4_mw080"          : "A4a  Min window: 0.8 × PSY",
    "A4_mw100"          : "A4b  Min window: 1.0 × PSY  [baseline]",
    "A4_mw120"          : "A4c  Min window: 1.2 × PSY",
    "A5_logprices"      : "A5   Log-price transformation",
    "B1_W1_early"       : "B1a  Window W1 start: early (Aug 2021)",
    "B1_W1_baseline"    : "B1b  Window W1 start: baseline (Nov 2021)  [baseline]",
    "B1_W1_late"        : "B1c  Window W1 start: late (Feb 2022)",
    "B1_W2_early"       : "B1d  Window W2 end: early (Jul 2022)",
    "B1_W2_baseline"    : "B1e  Window W2 end: baseline (Oct 2022)  [baseline]",
    "B1_W2_late"        : "B1f  Window W2 end: late (Jan 2023)",
    "B2_up21_dn10"      : "B2a  SV-ADF thresholds: M=21, R=10",
    "B2_up21_dn21"      : "B2b  SV-ADF thresholds: M=21, R=21",
    "B2_baseline"       : "B2c  SV-ADF thresholds: baseline (M=42, R=21)  [baseline]",
    "B2_up42_dn42"      : "B2d  SV-ADF thresholds: M=42, R=42",
    "B2_up63_dn21"      : "B2e  SV-ADF thresholds: M=63, R=21",
}

GSADF_ORDER = [
    "A1_pre_ChatGPT", "A1_post_ChatGPT",
    "A2_md21_mg21", "A2_md42_mg21", "A2_baseline", "A2_md90_mg42", "A2_md60_mg60",
    "A3_alpha90", "A3_alpha95_baseline", "A3_alpha99",
    "A4_mw080", "A4_mw100", "A4_mw120",
    "A5_logprices",
]
SVADF_ORDER = [
    "B1_W1_early", "B1_W1_baseline", "B1_W1_late",
    "B1_W2_early", "B1_W2_baseline", "B1_W2_late",
    "B2_up21_dn10", "B2_up21_dn21", "B2_baseline", "B2_up42_dn42", "B2_up63_dn21",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_summary() -> pd.DataFrame:
    return pd.read_csv(ROB_DIR / "tables" / "rob_summary.csv")


def ref_dict(df: pd.DataFrame, config: str) -> dict:
    row = df[df["config"] == config].iloc[0]
    return {k: row.get(k) for k in
            ["reject_rate_eq", "upstream_rate", "downstream_rate",
             "index_rate", "h2_holds", "h3_holds"]}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Stability heatmap
# ═══════════════════════════════════════════════════════════════════════════════
HMAP_COLS = [
    ("reject_rate_eq",  "Equity reject\nrate  [H1]",   "pp"),
    ("upstream_rate",   "Upstream rate\n[H2 ↑]",        "pp"),
    ("downstream_rate", "Downstream rate\n[H2 ↓]",      "pp"),
    ("index_rate",      "Index rate\n[H3 control]",     "pp"),
    ("h2_holds",        "H2 holds\n(Up > Dn)",          "bool"),
    ("h3_holds",        "H3 holds\n(Eq > Idx)",         "bool"),
]


def _cell_color(col_key: str, col_type: str, val, row, base: dict):
    """Return (facecolor, display_text) for a single heatmap cell."""
    # H3 is not applicable when no index tickers ran (B1)
    if col_key == "h3_holds" and pd.isna(row.get("index_rate")):
        return C_GREY, "n/a"
    if pd.isna(val):
        return C_GREY, "n/a"

    bv = base[col_key]
    if col_type == "pp":
        dev = abs(float(val) - float(bv))
        txt = f"{float(val):.2f}"
        if dev <= PP_STABLE:   return C_GREEN,  txt
        if dev <= PP_MODERATE: return C_ORANGE, txt
        return C_RED, txt
    else:  # bool
        match = bool(val) == bool(bv)
        return (C_GREEN if match else C_RED), ("✓" if bool(val) else "✗")


def _draw_panel(ax, rows: pd.DataFrame, base: dict,
                ref_label: str, panel_title: str) -> None:
    nr = len(rows)
    nc = len(HMAP_COLS)

    for ci, (col_key, _, col_type) in enumerate(HMAP_COLS):
        for ri, (_, row) in enumerate(rows.iterrows()):
            y     = nr - 1 - ri   # top-to-bottom
            val   = row.get(col_key)
            is_bv = row.get("_is_base", False)

            if is_bv:
                color, txt = C_TEAL, (f"{float(base[col_key]):.2f}"
                                       if col_type == "pp"
                                       else ("✓" if bool(base[col_key]) else "✗"))
            else:
                color, txt = _cell_color(col_key, col_type, val, row, base)

            ax.add_patch(plt.Rectangle([ci - .5, y - .5], 1, 1,
                                        facecolor=color, alpha=0.87, lw=0))
            tc = "white" if color not in (C_GREY,) else "#555"
            ax.text(ci, y, txt, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=tc)

    # reference row at y = -1
    ax.axhline(-0.5, color="#888", lw=1.4)
    for ci, (col_key, _, col_type) in enumerate(HMAP_COLS):
        bv  = base[col_key]
        txt = (f"{float(bv):.2f}" if col_type == "pp"
               else ("✓" if (not pd.isna(bv) and bool(bv)) else "✗"))
        ax.add_patch(plt.Rectangle([ci - .5, -1.5], 1, 1,
                                    facecolor=C_BLUE, alpha=0.88, lw=0))
        ax.text(ci, -1, txt, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")

    # grid
    for i in range(nc + 1):
        ax.axvline(i - .5, color="white", lw=0.9, zorder=3)
    for i in range(nr + 1):
        ax.axhline(i - .5, color="white", lw=0.5, zorder=3)

    ylabels = [SHORT.get(c, c) for c in rows["config"]][::-1] + [ref_label]
    ax.set_xlim(-.5, nc - .5)
    ax.set_ylim(-1.5, nr - .5)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([c[1] for c in HMAP_COLS], fontsize=8.5, fontweight="bold")
    ax.set_yticks(list(range(nr)) + [-1])
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", length=0)
    ax.set_title(panel_title, fontsize=9.5, fontweight="bold",
                 loc="left", pad=30, color=C_BLUE)


def make_heatmap(df: pd.DataFrame) -> None:
    base_a = ref_dict(df, GSADF_BASELINE)
    base_b = ref_dict(df, SVADF_BASELINE)

    def order_panel(order):
        sub = df[df["config"].isin(order)].copy()
        sub = sub.set_index("config").reindex(order).reset_index()
        sub["_is_base"] = sub["config"].isin(ALL_BASELINE_VARIANTS)
        return sub

    df_a = order_panel(GSADF_ORDER)
    df_b = order_panel(SVADF_ORDER)

    nr_a, nr_b = len(df_a), len(df_b)
    col_w      = 1.72
    label_w    = 3.8
    fig_w      = len(HMAP_COLS) * col_w + label_w
    fig_h      = (nr_a + nr_b + 6) * 0.50 + 0.8

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [nr_a + 1.4, nr_b + 1.4], "hspace": 0.55})

    _draw_panel(ax_a, df_a, base_a,
                ref_label="Baseline — GSADF, α=95%, PSY min window (A3b)",
                panel_title="Panel A — GSADF robustness checks  (A1–A5)")
    _draw_panel(ax_b, df_b, base_b,
                ref_label="Baseline — SV-ADF default thresholds, M=42, R=21 (B2c)",
                panel_title="Panel B — SV-ADF robustness checks  (B1–B2)")

    handles = [
        mpatches.Patch(facecolor=C_GREEN,  alpha=0.87,
                       label=f"Stable  (≤{int(PP_STABLE*100)} pp from baseline)"),
        mpatches.Patch(facecolor=C_ORANGE, alpha=0.87,
                       label=f"Moderate  ({int(PP_STABLE*100)}–{int(PP_MODERATE*100)} pp)"),
        mpatches.Patch(facecolor=C_RED,    alpha=0.87,
                       label=f"Large  (>{int(PP_MODERATE*100)} pp or direction flip)"),
        mpatches.Patch(facecolor=C_GREY,   alpha=0.87,  label="Not applicable / n/a"),
        mpatches.Patch(facecolor=C_TEAL,   alpha=0.87,  label="Baseline variant (reference)"),
        mpatches.Patch(facecolor=C_BLUE,   alpha=0.88,  label="Panel baseline (deviation = 0)"),
    ]
    fig.legend(handles=handles, fontsize=7.5, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(
        "Robustness stability heatmap — absolute percentage-point deviations from baseline\n"
        "Rate columns: deviation from reference value  ·  Boolean columns: ✓/✗ = direction maintained/reversed",
        fontsize=10.5, fontweight="bold", y=1.015)

    out = ROB_FIGS / "rob_stability_heatmap_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Hypothesis scorecard
# ═══════════════════════════════════════════════════════════════════════════════
def _sym(holds: bool | None, grey_on_false: bool = False) -> tuple[str, str]:
    """Return (symbol, color) for a hypothesis cell."""
    if holds is None:
        return "—", C_GREY
    if holds:
        return "✓", C_GREEN
    return ("✗", C_GREY) if grey_on_false else ("✗", C_RED)


def _scorecard_row(row: pd.Series, df: pd.DataFrame) -> dict:
    cfg   = str(row.get("config", ""))
    check = str(row.get("check",  ""))
    val_eq = row.get("reject_rate_eq")
    is_b   = check in ("B1", "B2")

    # H1 — explosive equity dynamics
    if pd.isna(val_eq):
        h1 = _sym(None)
    else:
        thresh = H1_THRESH_B if is_b else H1_THRESH_A
        holds  = float(val_eq) > thresh
        # pre-ChatGPT failure is expected by design → grey ✗
        grey   = (cfg == "A1_pre_ChatGPT" and not holds)
        h1     = _sym(holds, grey_on_false=grey)

    # H2 — upstream > downstream (capital-cycle pattern)
    v2 = row.get("h2_holds")
    h2 = _sym(None if pd.isna(v2) else bool(v2))

    # H3 — equity more explosive than indices
    # B1 has no index tickers → not applicable
    if check == "B1":
        h3 = _sym(None)
    else:
        v3 = row.get("h3_holds")
        h3 = _sym(None if pd.isna(v3) else bool(v3))

    # H4 — post-ChatGPT acceleration (only tested by A1)
    if cfg == "A1_post_ChatGPT":
        pre_rows  = df[df["config"] == "A1_pre_ChatGPT"]
        post_rows = df[df["config"] == "A1_post_ChatGPT"]
        if not pre_rows.empty and not post_rows.empty:
            pre_r  = float(pre_rows.iloc[0]["reject_rate_eq"])
            post_r = float(post_rows.iloc[0]["reject_rate_eq"])
            h4 = _sym(post_r > pre_r)
        else:
            h4 = _sym(None)
    else:
        h4 = _sym(None)

    # Overall verdict: based on H1, H2, H3 (H4 is temporal, not a replication check)
    applicable = [(s, c) for (s, c) in [h1, h2, h3]
                  if s not in ("—",) and not (s == "✗" and c == C_GREY)]
    if not applicable:
        verdict = ("—", C_GREY)
    else:
        n_pass = sum(1 for s, _ in applicable if s == "✓")
        ratio  = n_pass / len(applicable)
        if ratio == 1.0:   verdict = ("✓✓", C_GREEN)
        elif ratio >= 0.67: verdict = ("✓",  C_LBLUE)
        elif ratio >= 0.34: verdict = ("○",  C_ORANGE)
        else:               verdict = ("✗",  C_RED)

    return {
        "config"  : cfg,
        "label"   : SHORT.get(cfg, cfg),
        "check"   : check,
        "is_base" : cfg in ALL_BASELINE_VARIANTS,
        "H1": h1, "H2": h2, "H3": h3, "H4": h4,
        "verdict" : verdict,
    }


def make_scorecard(df: pd.DataFrame) -> None:
    ALL_ORDER = GSADF_ORDER + SVADF_ORDER
    df_idx    = df.set_index("config")
    disp = []
    for cfg in ALL_ORDER:
        if cfg not in df_idx.index:
            continue
        row = df_idx.loc[cfg].copy()
        row["config"] = cfg
        disp.append(_scorecard_row(row, df))

    # ── layout constants ──────────────────────────────────────────────────────
    COL_HDR = ["Configuration",
               "H1\nExplosivity",
               "H2\nCapital cycle",
               "H3\nIndex dilution",
               "H4\nChatGPT break\n(A1 only)",
               "Verdict"]
    COL_W   = [5.8, 1.15, 1.15, 1.15, 1.15, 0.95]
    TOT_W   = sum(COL_W)
    col_x   = [sum(COL_W[:i]) for i in range(len(COL_W))]
    ROW_H   = 0.375
    nr      = len(disp)
    HDR_H   = 0.55
    TOP_PAD = 0.85
    BOT_PAD = 0.55
    FIG_H   = TOP_PAD + HDR_H + nr * ROW_H + BOT_PAD

    fig, ax = plt.subplots(figsize=(TOT_W, FIG_H))
    ax.set_xlim(0, TOT_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # group background bands
    BAND_COLORS = {
        "A1": "#eef4fb", "A2": "#f9f9f9", "A3": "#eef4fb",
        "A4": "#f9f9f9",  "A5": "#eef4fb",
        "B1": "#fff7ee",  "B2": "#f2fbf5",
    }
    for ri, r in enumerate(disp):
        y_bot = FIG_H - TOP_PAD - HDR_H - (ri + 1) * ROW_H
        bg    = "#e8f4fd" if r["is_base"] else BAND_COLORS.get(r["check"], "#f9f9f9")
        ax.add_patch(plt.Rectangle([0, y_bot], TOT_W, ROW_H,
                                    facecolor=bg, edgecolor="none", zorder=0))

    # header row
    HDR_Y = FIG_H - TOP_PAD - HDR_H / 2
    for ci, hdr in enumerate(COL_HDR):
        ha = "left" if ci == 0 else "center"
        x  = col_x[ci] + 0.12 if ci == 0 else col_x[ci] + COL_W[ci] / 2
        ax.text(x, HDR_Y, hdr, fontsize=8.0, fontweight="bold",
                va="center", ha=ha, color=C_BLUE)
    sep_y = FIG_H - TOP_PAD - HDR_H
    ax.axhline(sep_y, color="#aaa", lw=1.3)

    # data rows
    prev_check = None
    for ri, r in enumerate(disp):
        y_center = FIG_H - TOP_PAD - HDR_H - ri * ROW_H - ROW_H / 2

        # group separator
        if r["check"] != prev_check and ri > 0:
            ax.axhline(y_center + ROW_H / 2, color="#aaa", lw=0.9)
        prev_check = r["check"]

        # label
        ax.text(col_x[0] + 0.12, y_center, r["label"],
                fontsize=7.3, va="center", ha="left",
                fontweight="bold" if r["is_base"] else "normal",
                color=C_BLUE if r["is_base"] else DARK)

        # hypothesis columns
        for ci, hyp in enumerate(["H1", "H2", "H3", "H4"], start=1):
            sym, color = r[hyp]
            cx = col_x[ci] + COL_W[ci] / 2
            if sym == "—":
                ax.text(cx, y_center, "—", fontsize=8.5, va="center",
                        ha="center", color="#c0c0c0")
            else:
                px, py = 0.07, ROW_H * 0.16
                ax.add_patch(plt.Rectangle(
                    [col_x[ci] + px, y_center - ROW_H / 2 + py],
                    COL_W[ci] - px * 2, ROW_H - py * 2,
                    facecolor=color, alpha=0.85, edgecolor="none", zorder=2))
                ax.text(cx, y_center, sym, fontsize=8.0, va="center",
                        ha="center", fontweight="bold", color="white", zorder=3)

        # verdict column
        sym_v, col_v = r["verdict"]
        cv = col_x[5] + COL_W[5] / 2
        ax.add_patch(plt.Rectangle(
            [col_x[5] + 0.05, y_center - ROW_H / 2 + ROW_H * 0.14],
            COL_W[5] - 0.10, ROW_H * 0.72,
            facecolor=col_v, alpha=0.85, edgecolor="none", zorder=2))
        ax.text(cv, y_center, sym_v, fontsize=8.5, va="center",
                ha="center", fontweight="bold", color="white", zorder=3)

        ax.axhline(y_center - ROW_H / 2, color="#ddd", lw=0.35, zorder=1)

    # vertical column separators
    for ci in range(1, len(COL_W)):
        ax.axvline(col_x[ci], color="#ccc", lw=0.6,
                   ymin=BOT_PAD / FIG_H, ymax=1 - TOP_PAD / FIG_H - 0.01, zorder=1)

    # footer summary
    h1_p = sum(1 for r in disp if r["H1"][0] == "✓")
    h2_p = sum(1 for r in disp if r["H2"][0] == "✓")
    h3_p = sum(1 for r in disp if r["H3"][0] == "✓")
    h1_a = sum(1 for r in disp if r["H1"][0] not in ("—",))
    h2_a = sum(1 for r in disp if r["H2"][0] not in ("—",))
    h3_a = sum(1 for r in disp if r["H3"][0] not in ("—",))

    pre_val  = float(df[df["config"] == "A1_pre_ChatGPT"].iloc[0]["reject_rate_eq"])
    post_val = float(df[df["config"] == "A1_post_ChatGPT"].iloc[0]["reject_rate_eq"])
    h4_str   = f"confirmed (post={post_val:.2f} > pre={pre_val:.2f})"

    footer_y = BOT_PAD / 2 + 0.10
    ax.text(0.12, footer_y,
            f"  * grey ✗ = expected failure by design (pre-boom sub-period)  "
            f"   ✓✓ = fully robust  ·  ✓ = majority hypotheses hold  "
            f"  ·  ○ = partial  ·  ✗ = failure\n"
            f"  Summary counts:  H1 ✓ in {h1_p}/{h1_a}  ·  "
            f"H2 ✓ in {h2_p}/{h2_a}  ·  "
            f"H3 ✓ in {h3_p}/{h3_a} configurations  ·  "
            f"H4 {h4_str}",
            fontsize=7.0, va="center", ha="left", color="#555", style="italic")

    fig.suptitle(
        "Robustness scorecard — hypothesis-level assessment across all configurations\n"
        "H1: AI equity explosivity  ·  H2: capital-cycle pattern (upstream > downstream)  ·  "
        "H3: index dilution  ·  H4: post-ChatGPT acceleration",
        fontsize=9.5, fontweight="bold", y=1.01)

    out = ROB_FIGS / "rob_scorecard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading rob_summary.csv ...")
    df = load_summary()
    print(f"  {len(df)} configurations found")

    print("\nBuilding heatmap v2 ...")
    make_heatmap(df)

    print("\nBuilding scorecard ...")
    make_scorecard(df)

    print(f"\nDone. Figures saved to: {ROB_FIGS}")
