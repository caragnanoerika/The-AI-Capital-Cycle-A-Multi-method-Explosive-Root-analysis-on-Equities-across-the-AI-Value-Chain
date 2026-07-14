"""
Step 12 — GSADF results table for individual equities.

Generates a publication-ready table covering all equity tickers
(segments 1–11, i.e. segment label does NOT start with C./0./P.).

For each ticker the table reports:
  Segment | Ticker | Company | T | GSADF stat | CV 95% | p-value | Decision | # Episodes | Episode periods

p-value brackets are derived from the three MC quantiles stored in
data/mc_cache/global_critical_values.csv (q900, q950, q990).

Output
------
    outputs/tables/gsadf/gsadf_equity_results.csv   ← copy-paste into Excel/Word
    outputs/tables/gsadf/gsadf_equity_results.tex   ← paste directly into LaTeX

Usage
-----
    python scripts/12_gsadf_equity_table.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import settings
from src.io.results import load_gsadf


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_equity(segment: str) -> bool:
    """Equity segments are numbered 1.–11.; commodities start with C., indices 0., planned P."""
    return segment[:2].strip(".").isdigit() and not segment.startswith(("C.", "0.", "P."))


def _load_mc_index() -> pd.DataFrame:
    mc = pd.read_csv(settings.MC_GLOBAL_INDEX_FILE)
    # Keep only full-precision runs (nrep=999, seed=42)
    mc = mc[(mc["nrep"] == settings.GSADF_MC_REPS) & (mc["seed"] == settings.SEED)]
    return mc.set_index(["T", "min_window"])


def _pvalue_bracket(stat: float, T: int, min_window: int, mc: pd.DataFrame) -> str:
    try:
        row = mc.loc[(T, min_window)]
        q99, q95, q90 = row["gsadf_q990"], row["gsadf_q950"], row["gsadf_q900"]
        if stat > q99:  return "< 0.01"
        if stat > q95:  return "< 0.05"
        if stat > q90:  return "< 0.10"
        return "> 0.10"
    except KeyError:
        return "n/a"


def _fmt_episodes(episodes: list[dict]) -> str:
    if not episodes:
        return "—"
    parts = []
    for ep in episodes:
        s = ep["start"][:7]   # YYYY-MM
        e = ep["end"][:7]
        parts.append(f"{s} → {e}")
    return "; ".join(parts)


def _latex_escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#")
             .replace("→", r"$\to$")
             .replace("—", r"---"))


# ── build table ───────────────────────────────────────────────────────────────

def build_gsadf_equity_table() -> pd.DataFrame:
    mc = _load_mc_index()
    rows = []

    for tk in settings.ALL_TICKERS:
        seg, name = settings.SEGMENT_OF.get(tk, ("?", tk))
        if not _is_equity(seg):
            continue

        gs = load_gsadf(tk)
        if gs is None:
            # Still include the ticker so the reader knows it was in the universe
            rows.append({
                "Segment": seg, "Ticker": tk, "Company": name,
                "T": "—", "GSADF stat": "—", "CV (95%)": "—",
                "p-value": "—", "Decision": "—",
                "# Episodes": "—", "Episode periods": "—",
            })
            continue

        g = gs["summary"]
        stat    = g["statistic"]
        cv      = g["cv_value"]
        T       = g["T"]
        mw      = g["min_window"]
        reject  = g["reject"]
        # PSY (2015): date-stamping is only valid conditional on global rejection.
        # Suppress episodes for tickers that fail the global null — pointwise
        # cv_path crossings can occur even when the supremum statistic does not
        # clear the higher supremum CV, producing spurious local episodes.
        episodes = g["episodes"] if reject else []
        n_ep    = len(episodes)
        ep_str  = _fmt_episodes(episodes)
        pval    = _pvalue_bracket(stat, T, mw, mc)

        rows.append({
            "Segment":        seg,
            "Ticker":         tk,
            "Company":        name,
            "T":              T,
            "GSADF stat":     round(stat, 3),
            "CV (95%)":       round(cv,   3),
            "p-value":        pval,
            "Decision":       "Reject" if reject else "Fail to reject",
            "# Episodes":     n_ep,
            "Episode periods": ep_str,
        })

    return pd.DataFrame(rows)


# ── LaTeX renderer ────────────────────────────────────────────────────────────

def _to_latex(df: pd.DataFrame) -> str:
    col_fmt = "llp{3.8cm}rrrllrp{4cm}"   # one format token per column
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{GSADF test results — individual equities (2020--2026, MC 999 reps, seed 42, 95\% critical value)}",
        r"\label{tab:gsadf_equity}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
    ]

    headers = [_latex_escape(c) for c in df.columns]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    prev_seg = None
    for _, row in df.iterrows():
        seg = row["Segment"]
        if seg != prev_seg and prev_seg is not None:
            lines.append(r"\midrule")
        prev_seg = seg

        cells = []
        for col, val in row.items():
            s = str(val)
            # Bold the stat if it rejects
            if col == "GSADF stat" and row["Decision"] == "Reject":
                s = r"\textbf{" + s + "}"
            # Colour the decision
            if col == "Decision":
                s = (r"\textcolor{red!60!black}{Reject}"
                     if val == "Reject"
                     else r"\textcolor{gray}{Fail to reject}")
            cells.append(_latex_escape(s) if col not in ("GSADF stat",) else s)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item \textit{Notes:} T = sample size (trading days). "
        r"GSADF stat = supremum of the backward-SADF sequence. "
        r"CV (95\%) = Monte Carlo 95th-percentile critical value. "
        r"p-value brackets are derived from the q90/q95/q99 MC quantiles. "
        r"Episode periods are date-stamped using the BSADF $>$ CV crossing rule "
        r"(min.\ duration 60 days, merge gap 42 days).",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings.TABLES_GSADF_DIR.mkdir(parents=True, exist_ok=True)

    df = build_gsadf_equity_table()

    # CSV
    csv_path = settings.TABLES_GSADF_DIR / "gsadf_equity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV  → {csv_path}")

    # LaTeX
    tex_path = settings.TABLES_GSADF_DIR / "gsadf_equity_results.tex"
    tex_path.write_text(_to_latex(df), encoding="utf-8")
    print(f"LaTeX → {tex_path}")

    # Quick preview in terminal
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
