"""
08_robustness.py
Orchestrate all robustness checks for the AI bubble-detection analysis.

All outputs are written to  outputs/robustness/  and are therefore
clearly separate from the main-analysis results in outputs/tables/ and
outputs/figures/.  No cached data in data/results/ is ever overwritten.

Checks
------
A1  GSADF sub-period split       pre-ChatGPT (2020-01-01..2022-10-31)
                                  post-ChatGPT (2022-11-01..2026-05-01)
A2  GSADF episode-filter grid    min_duration x merge_gap sweep
A3  GSADF significance level     alpha = 90%, 95% (baseline), 99%
A4  GSADF min-window variants    PSY mw x {0.80, 1.00, 1.20}
A5  GSADF log prices             re-run on log(price), same MC cache
B1  SV-ADF window boundary       ChatGPT split shifted +/- 3 months
B2  SV-ADF episode thresholds    SV_MIN_UP x SV_MIN_DOWN grid

S   Summary table + stability heatmap  (always built at the end)

Usage
-----
  python scripts/08_robustness.py                    # run all checks
  python scripts/08_robustness.py --checks A2 A3 B2  # run subset
  python scripts/08_robustness.py --skip-mc           # skip A1 / A4

Feeds into: scripts/08b_robustness_figures.py, which reads rob_summary.csv
and expects the exact `config` tags this script produces — if you change a
grid's tags here, update SHORT / *_ORDER in 08b to match.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import settings
from src.io.data        import load_or_download_prices
from src.io.results     import load_gsadf, load_svadf, list_svadf_windows
from src.methods.gsadf  import gsadf as run_gsadf, minimum_window
from src.methods.episodes import detect_bsadf_episodes, detect_sv_episode
from src.methods.svadf  import svadf as run_svadf_method
from src.monte_carlo    import get_critical_values

# ── output paths (separate from main analysis) ────────────────────────────────
ROB_DIR    = PROJECT_ROOT / "outputs" / "robustness"
ROB_TABLES = ROB_DIR / "tables"
ROB_FIGS   = ROB_DIR / "figures"

CHATGPT = pd.Timestamp("2022-11-01")

# ── segment helpers ───────────────────────────────────────────────────────────
def _seg_class(seg: str) -> str:
    if seg.startswith("0."):  return "index"
    if seg.startswith("C."):  return "commodity"
    if seg.startswith("P."):  return "planned"
    try:
        n = int(seg.split(".")[0])
        if n <= 4:  return "upstream"
        if n <= 8:  return "midstream"
        return "downstream"
    except ValueError:
        return "other"

def _is_equity(seg: str) -> bool:
    return _seg_class(seg) in ("upstream", "midstream", "downstream")

# ── canonical baseline metrics from cached GSADF results ─────────────────────
def _baseline_metrics() -> pd.DataFrame:
    rows = []
    for tk in settings.ALL_TICKERS:
        r = load_gsadf(tk)
        if r is None:
            continue
        s = r["summary"]
        seg, _ = settings.SEGMENT_OF.get(tk, ("?", ""))
        rows.append({
            "ticker"    : tk,
            "segment"   : seg,
            "seg_class" : _seg_class(seg),
            "T"         : s.get("T", 0),
            "mw"        : s.get("min_window", minimum_window(s.get("T", 0))),
            "stat"      : s["statistic"],
            "reject_95" : s["reject"],
            "n_ep_95"   : len(s.get("episodes", [])),
        })
    return pd.DataFrame(rows)

# ── generic GSADF summary metrics from a rows list ────────────────────────────
def _metrics(rows: list[dict], config: str, alpha_key: str = "reject") -> dict:
    eq  = [r for r in rows if _is_equity(r.get("segment", ""))]
    idx = [r for r in rows if _seg_class(r.get("segment", "")) == "index"]
    up  = [r for r in eq  if _seg_class(r.get("segment", "")) == "upstream"]
    dn  = [r for r in eq  if _seg_class(r.get("segment", "")) == "downstream"]
    _rate = lambda lst, k: sum(1 for x in lst if x.get(k)) / len(lst) if lst else float("nan")
    return {
        "config"          : config,
        "n_equity"        : len(eq),
        "reject_rate_eq"  : _rate(eq,  alpha_key),
        "upstream_rate"   : _rate(up,  alpha_key),
        "downstream_rate" : _rate(dn,  alpha_key),
        "h2_holds"        : (_rate(up, alpha_key) > _rate(dn, alpha_key)),
        "index_rate"      : _rate(idx, alpha_key),
        "h3_holds"        : (_rate(idx, alpha_key) < _rate(eq, alpha_key)),
        "mean_episodes"   : (sum(r.get("n_ep", 0) for r in eq) / len(eq)) if eq else float("nan"),
    }

# ── run GSADF on a price series (used by A1, A4, A5) ─────────────────────────
def _gsadf_on_series(tk: str, s: pd.Series,
                     mw_override: int | None = None,
                     min_dur: int | None = None,
                     merge_gap: int | None = None) -> dict | None:
    if len(s) < 60:
        return None
    T   = len(s)
    mw  = mw_override or minimum_window(T)
    md  = min_dur  or settings.EPISODE_MIN_DURATION
    mg  = merge_gap or settings.EPISODE_MERGE_GAP
    seg, _ = settings.SEGMENT_OF.get(tk, ("?", ""))
    g   = run_gsadf(s.to_numpy(), min_window=mw)
    cv  = get_critical_values(T=T, min_window=mw,
                              nrep=settings.GSADF_MC_REPS,
                              seed=settings.SEED,
                              quantiles=(0.90, 0.95, 0.99),
                              verbose=False)
    dt  = s.index[mw - 1:]
    return {
        "ticker"    : tk,
        "segment"   : seg,
        "T"         : T,
        "mw"        : mw,
        "stat"      : g["statistic"],
        "reject_90" : g["statistic"] > cv["gsadf"][0.90],
        "reject_95" : g["statistic"] > cv["gsadf"][0.95],
        "reject_99" : g["statistic"] > cv["gsadf"][0.99],
        "n_ep_90"   : len(detect_bsadf_episodes(g["path"], cv["bsadf_path"][0.90], dt, md, mg)),
        "n_ep_95"   : len(detect_bsadf_episodes(g["path"], cv["bsadf_path"][0.95], dt, md, mg)),
        "n_ep_99"   : len(detect_bsadf_episodes(g["path"], cv["bsadf_path"][0.99], dt, md, mg)),
        "n_ep"      : len(detect_bsadf_episodes(g["path"], cv["bsadf_path"][0.95], dt, md, mg)),
        "reject"    : g["statistic"] > cv["gsadf"][0.95],
    }

# ═════════════════════════════════════════════════════════════════════════════
# A1 — GSADF sub-period split
# ═════════════════════════════════════════════════════════════════════════════
def run_A1(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n[A1] GSADF sub-period split ...")
    periods = {
        "A1_pre_ChatGPT"  : (settings.FIXED_START, "2022-10-31"),
        "A1_post_ChatGPT" : ("2022-11-01",          settings.FIXED_END),
    }
    all_rows = []
    for tag, (start, end) in periods.items():
        t0 = time.time()
        print(f"  {tag}: {start} to {end}")
        rows = []
        for tk in settings.ALL_TICKERS:
            if tk not in prices.columns:
                continue
            s = prices[tk].dropna().loc[start:end]
            r = _gsadf_on_series(tk, s)
            if r:
                r["period"] = tag
                rows.append(r)
        print(f"    {len(rows)} tickers  ({time.time()-t0:.0f}s)")
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)

# ═════════════════════════════════════════════════════════════════════════════
# A2 — GSADF episode filter grid  (fast — reuses cached paths)
# ═════════════════════════════════════════════════════════════════════════════
def run_A2() -> pd.DataFrame:
    print("\n[A2] GSADF episode-filter grid ...")
    grid = [
        (21,  21, "A2_md21_mg21"),
        (42,  21, "A2_md42_mg21"),
        (60,  42, "A2_baseline"),   # settings defaults
        (90,  42, "A2_md90_mg42"),
        (60,  60, "A2_md60_mg60"),
    ]
    summary_rows = []
    for md, mg, tag in grid:
        rows = []
        for tk in settings.ALL_TICKERS:
            r = load_gsadf(tk)
            if r is None:
                continue
            paths = r["paths"]
            if paths is None:
                continue
            seg, _ = settings.SEGMENT_OF.get(tk, ("?", ""))
            dt  = paths.index
            eps = detect_bsadf_episodes(
                paths["bsadf_stat"].to_numpy(),
                paths["cv_path"].to_numpy(),
                dt, min_duration=md, merge_gap=mg)
            rows.append({
                "ticker": tk, "segment": seg,
                "n_ep": len(eps), "reject": len(eps) > 0,
                "min_duration": md, "merge_gap": mg, "config": tag,
            })
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        m = _metrics(rows, tag)
        m["min_duration"] = md
        m["merge_gap"]    = mg
        summary_rows.append(m)
        print(f"  {tag}: equity reject={m['reject_rate_eq']:.2f}  "
              f"mean_ep={m['mean_episodes']:.2f}")
    return pd.DataFrame(summary_rows)

# ═════════════════════════════════════════════════════════════════════════════
# A3 — GSADF significance level  (fast — CVs already in MC cache)
# ═════════════════════════════════════════════════════════════════════════════
def run_A3() -> pd.DataFrame:
    print("\n[A3] GSADF significance-level variants ...")
    baseline = _baseline_metrics()
    alphas   = {0.90: "A3_alpha90", 0.95: "A3_alpha95_baseline", 0.99: "A3_alpha99"}
    summary  = []
    for q, tag in alphas.items():
        rows = []
        for _, row in baseline.iterrows():
            tk, T, mw = row["ticker"], int(row["T"]), int(row["mw"])
            cv = get_critical_values(T=T, min_window=mw,
                                     nrep=settings.GSADF_MC_REPS,
                                     seed=settings.SEED,
                                     quantiles=(q,), verbose=False)
            cv_val  = cv["gsadf"][q]
            cv_path = cv["bsadf_path"][q]
            r_load  = load_gsadf(tk)
            seg, _  = settings.SEGMENT_OF.get(tk, ("?", ""))
            if r_load is None or r_load["paths"] is None:
                continue
            dt  = r_load["paths"].index
            bsp = r_load["paths"]["bsadf_stat"].to_numpy()
            eps = detect_bsadf_episodes(bsp, cv_path, dt,
                                        settings.EPISODE_MIN_DURATION,
                                        settings.EPISODE_MERGE_GAP)
            rows.append({
                "ticker": tk, "segment": seg,
                "stat": row["stat"], "cv": cv_val,
                "reject": row["stat"] > cv_val,
                "n_ep": len(eps), "config": tag, "alpha": q,
            })
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        m = _metrics(rows, tag)
        m["alpha"] = q
        summary.append(m)
        print(f"  {tag}: equity reject={m['reject_rate_eq']:.2f}")
    return pd.DataFrame(summary)

# ═════════════════════════════════════════════════════════════════════════════
# A4 — GSADF min-window variants  (needs MC for new mw values)
# ═════════════════════════════════════════════════════════════════════════════
def run_A4(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n[A4] GSADF min-window variants (+/-20% of PSY) ...")
    variants = {
        "A4_mw080": 0.80,
        "A4_mw100": 1.00,   # baseline
        "A4_mw120": 1.20,
    }
    summary = []
    for tag, scale in variants.items():
        t0   = time.time()
        rows = []
        for tk in settings.ALL_TICKERS:
            if tk not in prices.columns:
                continue
            s  = prices[tk].dropna()
            T  = len(s)
            mw = max(10, int(minimum_window(T) * scale))
            r  = _gsadf_on_series(tk, s, mw_override=mw)
            if r:
                r["mw_scale"] = scale
                r["config"]   = tag
                rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        m = _metrics(rows, tag)
        m["mw_scale"] = scale
        summary.append(m)
        print(f"  {tag} (scale={scale}): equity reject={m['reject_rate_eq']:.2f}  "
              f"({time.time()-t0:.0f}s)")
    return pd.DataFrame(summary)

# ═════════════════════════════════════════════════════════════════════════════
# A5 — GSADF log prices  (no new MC — T and mw unchanged)
# ═════════════════════════════════════════════════════════════════════════════
def run_A5(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n[A5] GSADF log-price series ...")
    t0   = time.time()
    rows = []
    for tk in settings.ALL_TICKERS:
        if tk not in prices.columns:
            continue
        s_raw = prices[tk].dropna()
        if (s_raw <= 0).any():
            continue
        s_log = np.log(s_raw)
        r     = _gsadf_on_series(tk, s_log)
        if r:
            r["series"] = "log"
            r["config"] = "A5_logprices"
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ROB_TABLES / "rob_A5_logprices.csv", index=False, encoding="utf-8-sig")
    m = _metrics(rows, "A5_logprices")
    print(f"  equity reject={m['reject_rate_eq']:.2f}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame([m])

# ═════════════════════════════════════════════════════════════════════════════
# B1 — SV-ADF window boundary shift  (+/- 3 months on W1 and W2)
# ═════════════════════════════════════════════════════════════════════════════
def run_B1(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n[B1] SV-ADF window boundary shift ...")
    windows = {
        "B1_W1_early"    : ("2021-08-01", "2026-05-01"),
        "B1_W1_baseline" : (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
        "B1_W1_late"     : ("2022-02-01", "2026-05-01"),
        "B1_W2_early"    : (settings.SVADF_PRE_GPT_START, "2022-07-31"),
        "B1_W2_baseline" : (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
        "B1_W2_late"     : (settings.SVADF_PRE_GPT_START, "2023-01-31"),
    }
    summary = []
    equity_tickers = [tk for tk in settings.ALL_TICKERS
                      if _is_equity(settings.SEGMENT_OF.get(tk, ("?",))[0])]
    for tag, (w_start, w_end) in windows.items():
        rows = []
        for tk in equity_tickers:
            if tk not in prices.columns:
                continue
            s = prices[tk].dropna()
            sw = s.loc[w_start:w_end]
            if len(sw) < 100:
                continue
            seg, _ = settings.SEGMENT_OF.get(tk, ("?", ""))
            res    = run_svadf_method(sw.to_numpy())
            if res is None:
                continue
            ep = detect_sv_episode(
                res["coef_stat"], res["orig_thr"], res["screen_thr"], res["coll_thr"],
                dates=sw.index,
                M=settings.SV_MIN_UP, R=settings.SV_MIN_DOWN,
                bridge_days=settings.SV_BRIDGE_DAYS)
            rows.append({
                "ticker": tk, "segment": seg,
                "has_ep": ep is not None,
                "reject": ep is not None,
                "n_ep": int(ep is not None),
                "config": tag, "w_start": w_start, "w_end": w_end,
            })
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        m = _metrics(rows, tag, alpha_key="reject")
        m["w_start"] = w_start
        m["w_end"]   = w_end
        summary.append(m)
        eq_rate = sum(r["has_ep"] for r in rows) / len(rows) if rows else float("nan")
        print(f"  {tag} ({w_start}..{w_end}): episode rate={eq_rate:.2f}")
    return pd.DataFrame(summary)

# ═════════════════════════════════════════════════════════════════════════════
# B2 — SV-ADF episode threshold grid  (fast — reuses cached paths)
# ═════════════════════════════════════════════════════════════════════════════
def run_B2() -> pd.DataFrame:
    """
    Sweep SV_MIN_UP (M) x SV_MIN_DOWN (R) around the settings.py baseline,
    holding SV_BRIDGE_DAYS fixed (bridge sensitivity isn't this check's concern
    — B1 covers window-boundary sensitivity separately). Baseline is derived
    from settings.SV_MIN_UP/SV_MIN_DOWN rather than hardcoded so this grid
    can't silently drift out of sync with the live pipeline defaults again.
    """
    print("\n[B2] SV-ADF episode-threshold grid ...")
    M0, R0 = settings.SV_MIN_UP, settings.SV_MIN_DOWN   # baseline = 42, 21
    grid = [
        (M0 // 2,     R0 // 2,      f"B2_up{M0//2}_dn{R0//2}"),
        (M0 // 2,     R0,           f"B2_up{M0//2}_dn{R0}"),
        (M0,          R0,           "B2_baseline"),
        (M0,          M0,           f"B2_up{M0}_dn{M0}"),
        (M0 + R0,     R0,           f"B2_up{M0+R0}_dn{R0}"),
    ]
    bridge = settings.SV_BRIDGE_DAYS
    w1 = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
    summary = []
    for M, R, tag in grid:
        rows = []
        for tk in settings.ALL_TICKERS:
            r = load_svadf(tk, *w1)
            if r is None or r["paths"] is None:
                continue
            seg, _ = settings.SEGMENT_OF.get(tk, ("?", ""))
            p      = r["paths"]
            # screen_thr may be missing from older cached files; reconstruct if needed
            if "screen_thr" in p.columns:
                screen = p["screen_thr"].to_numpy()
            else:
                screen = p["orig_thr"].to_numpy() * 10  # log(τ) = 10 × (log(τ)/10)
            ep = detect_sv_episode(
                p["coef_stat"].to_numpy(),
                p["orig_thr"].to_numpy(),
                screen,
                p["coll_thr"].to_numpy(),
                dates=p.index,
                M=M, R=R, bridge_days=bridge)
            rows.append({
                "ticker": tk, "segment": seg,
                "has_ep": ep is not None,
                "reject": ep is not None,
                "n_ep": int(ep is not None),
                "config": tag, "min_up": M, "min_down": R,
            })
        df = pd.DataFrame(rows)
        df.to_csv(ROB_TABLES / f"rob_{tag}.csv", index=False, encoding="utf-8-sig")
        m = _metrics(rows, tag, alpha_key="reject")
        m["min_up"]   = M
        m["min_down"] = R
        summary.append(m)
        print(f"  {tag}: equity episode rate={m['reject_rate_eq']:.2f}")
    return pd.DataFrame(summary)

# ═════════════════════════════════════════════════════════════════════════════
# S — Summary table and stability heatmap
# ═════════════════════════════════════════════════════════════════════════════
def build_summary(results: dict) -> pd.DataFrame:
    """Combine all per-check summary DataFrames into one master table."""
    frames = []
    for check, df in results.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["check"] = check
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_stability_figure(summary: pd.DataFrame, baseline: dict) -> None:
    """
    Stability heatmap: rows = configurations, cols = key metrics.
    Green  = within 5pp of baseline
    Orange = 5-20pp deviation
    Red    = >20pp or direction reversal
    """
    metrics = ["reject_rate_eq", "upstream_rate", "downstream_rate",
               "index_rate", "mean_episodes"]
    labels  = ["Equity\nreject rate", "Upstream\nrate",
               "Downstream\nrate", "Index\nrate", "Mean\nepisodes"]

    rows = summary.dropna(subset=["reject_rate_eq"])

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.6 + 2, max(4, len(rows) * 0.55)))

    for ci, (col, lbl) in enumerate(zip(metrics, labels)):
        base = baseline.get(col, float("nan"))
        for ri, (_, row) in enumerate(rows.iterrows()):
            val = row.get(col, float("nan"))
            if pd.isna(val) or pd.isna(base) or base == 0:
                color = "#cccccc"
                txt   = "n/a"
            else:
                dev = abs(val - base) / (abs(base) + 1e-9)
                if dev <= 0.05:   color = "#52b788"   # green
                elif dev <= 0.20: color = "#f4a261"   # orange
                else:             color = "#e63946"   # red
                # direction reversal overrides
                if col in ("h2_holds", "h3_holds"):
                    if row.get(col) != baseline.get(col):
                        color = "#e63946"
                txt = f"{val:.2f}" if isinstance(val, float) else str(val)
            rect = plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1,
                                  facecolor=color, alpha=0.85)
            ax.add_patch(rect)
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")

    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows["config"].tolist(), fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    legend = [
        mpatches.Patch(color="#52b788", alpha=0.85, label="Stable (<5pp from baseline)"),
        mpatches.Patch(color="#f4a261", alpha=0.85, label="Moderate (5-20pp)"),
        mpatches.Patch(color="#e63946", alpha=0.85, label="Large (>20pp or reversal)"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right",
              bbox_to_anchor=(1.0, -0.12), ncol=3)

    plt.title("Robustness stability heatmap\n"
              "(deviations from baseline main-analysis results)",
              fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(ROB_FIGS / "rob_stability_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {ROB_FIGS / 'rob_stability_heatmap.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checks", nargs="+", default=["all"],
        help="Checks to run: A1 A2 A3 A4 A5 B1 B2  (default: all)")
    parser.add_argument(
        "--skip-mc", action="store_true",
        help="Skip checks that may require new Monte Carlo (A1, A4)")
    args = parser.parse_args()

    run_all  = "all" in args.checks
    selected = set(args.checks) if not run_all else {"A1","A2","A3","A4","A5","B1","B2"}
    if args.skip_mc:
        selected -= {"A1", "A4"}
        print("  [--skip-mc] Skipping A1 and A4")

    ROB_TABLES.mkdir(parents=True, exist_ok=True)
    ROB_FIGS.mkdir(parents=True,   exist_ok=True)

    # load prices once (used by A1, A4, A5, B1)
    needs_prices = selected & {"A1", "A4", "A5", "B1"}
    prices       = pd.DataFrame()
    if needs_prices:
        print("Loading cached prices ...")
        prices = load_or_download_prices(force=False)
        print(f"  {prices.shape[0]} dates x {prices.shape[1]} tickers")

    # baseline for stability comparison
    base_df = _baseline_metrics()
    eq_base = base_df[base_df["seg_class"].isin(["upstream","midstream","downstream"])]
    baseline = {
        "reject_rate_eq"  : eq_base["reject_95"].mean(),
        "upstream_rate"   : eq_base[eq_base["seg_class"]=="upstream"]["reject_95"].mean(),
        "downstream_rate" : eq_base[eq_base["seg_class"]=="downstream"]["reject_95"].mean(),
        "index_rate"      : base_df[base_df["seg_class"]=="index"]["reject_95"].mean(),
        "mean_episodes"   : eq_base["n_ep_95"].mean(),
    }
    print(f"\nBaseline (full sample, alpha=95%):")
    for k, v in baseline.items():
        print(f"  {k}: {v:.3f}")

    results = {}

    if "A1" in selected:
        r = run_A1(prices)
        results["A1"] = pd.DataFrame([
            _metrics(r[r["period"]==p].to_dict("records"), p, "reject")
            for p in r["period"].unique()
        ]) if not r.empty else None

    if "A2" in selected:
        results["A2"] = run_A2()

    if "A3" in selected:
        results["A3"] = run_A3()

    if "A4" in selected:
        results["A4"] = run_A4(prices)

    if "A5" in selected:
        results["A5"] = run_A5(prices)

    if "B1" in selected:
        results["B1"] = run_B1(prices)

    if "B2" in selected:
        results["B2"] = run_B2()

    # summary
    print("\n[S] Building summary ...")
    summary = build_summary(results)
    if not summary.empty:
        summary.to_csv(ROB_TABLES / "rob_summary.csv",
                       index=False, encoding="utf-8-sig")
        print(f"  Summary table: {len(summary)} configurations")
        make_stability_figure(summary, baseline)

    print("\nDone. All outputs in:", ROB_DIR)
    print("  Tables :", ROB_TABLES)
    print("  Figures:", ROB_FIGS)


if __name__ == "__main__":
    main()
