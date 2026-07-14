"""
Step 13 — SV-ADF results table for individual equities.

Generates a publication-ready table covering all equity tickers
(segments 1–11, same universe as script 12 for the GSADF table).

For each ticker the table reports one row with results for both windows:
  Segment | Ticker | Company
  | W1 T | W1 Detected | W1 Start | W1 End | W1 Duration | W1 Collapse
  | W2 T | W2 Detected | W2 Start | W2 End | W2 Duration | W2 Collapse

SV-ADF uses analytical thresholds (no Monte Carlo), so there are no
p-value brackets. Detection is binary: episode found or not.

Windows
-------
    W1  post-ChatGPT  SVADF_DEFAULT_START → SVADF_DEFAULT_END   (primary)
    W2  pre-ChatGPT   SVADF_PRE_GPT_START → SVADF_PRE_GPT_END   (comparison)

Collapse types
--------------
    bridge        collapsed within 90-day bridge window after origination
    post_bridge   collapsed via screen + R-run rule after bridge period
    end_of_sample episode extends to end of window (no detected collapse)
    —             no episode detected

Output
------
    outputs/tables/svadf/svadf_equity_results.csv
    outputs/tables/svadf/svadf_equity_results.tex

Usage
-----
    python scripts/13_svadf_equity_table.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import settings
from src.io.results import load_svadf


# ── constants ─────────────────────────────────────────────────────────────────

W1 = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
W2 = (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END)

W1_LABEL = f"{W1[0]} → {W1[1]}"
W2_LABEL = f"{W2[0]} → {W2[1]}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_equity(segment: str) -> bool:
    return segment[:2].strip(".").isdigit() and not segment.startswith(("C.", "0.", "P."))


def _fmt_date(d: str | None) -> str:
    return d[:7] if d else "—"   # YYYY-MM


def _fmt_ep(ep: dict | None) -> tuple[str, str, str, str]:
    """Return (detected, start, end, duration, collapse) strings."""
    if ep is None:
        return "No", "—", "—", "—", "—"
    start = _fmt_date(str(ep.get("start", "")))
    end   = _fmt_date(str(ep.get("end",   "")))
    dur   = str(ep.get("duration_days", "—"))
    ct    = ep.get("collapse_type", "—") or "—"
    return "Yes", start, end, dur, ct


# ── build table ───────────────────────────────────────────────────────────────

def build_svadf_equity_table() -> pd.DataFrame:
    rows = []

    for tk in settings.ALL_TICKERS:
        seg, name = settings.SEGMENT_OF.get(tk, ("?", tk))
        if not _is_equity(seg):
            continue

        r1 = load_svadf(tk, *W1)
        r2 = load_svadf(tk, *W2)

        def _parse(r):
            if r is None:
                return {"T": "—", "detected": "—", "start": "—",
                        "end": "—", "duration": "—", "collapse": "—"}
            s = r["summary"]
            det, start, end, dur, ct = _fmt_ep(s.get("episode"))
            return {"T": s.get("T", "—"), "detected": det,
                    "start": start, "end": end, "duration": dur, "collapse": ct}

        w1 = _parse(r1)
        w2 = _parse(r2)

        rows.append({
            "Segment":       seg,
            "Ticker":        tk,
            "Company":       name,
            # W1
            "W1 T":          w1["T"],
            "W1 Detected":   w1["detected"],
            "W1 Start":      w1["start"],
            "W1 End":        w1["end"],
            "W1 Duration":   w1["duration"],
            "W1 Collapse":   w1["collapse"],
            # W2
            "W2 T":          w2["T"],
            "W2 Detected":   w2["detected"],
            "W2 Start":      w2["start"],
            "W2 End":        w2["end"],
            "W2 Duration":   w2["duration"],
            "W2 Collapse":   w2["collapse"],
        })

    return pd.DataFrame(rows)


# ── LaTeX renderer ────────────────────────────────────────────────────────────

def _latex_escape(s: str) -> str:
    return (str(s)
            .replace("&",  r"\&")
            .replace("%",  r"\%")
            .replace("_",  r"\_")
            .replace("#",  r"\#")
            .replace("→",  r"$\to$")
            .replace("—",  r"---"))


def _to_latex(df: pd.DataFrame) -> str:
    # Column format: seg | ticker | company | W1×5 | W2×5
    col_fmt = "llp{3cm}r l cc r l  r l cc r l"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (r"\caption{SV-ADF (Sarkar \& Wells 2026) results --- individual equities. "
         r"W1 = post-ChatGPT window (" + W1_LABEL.replace("→", r"$\to$") + r"); "
         r"W2 = pre-ChatGPT window (" + W2_LABEL.replace("→", r"$\to$") + r"). "
         r"Detection is binary: analytical origination threshold log($\tau$)/10, "
         r"min.\ 60 consecutive steps. "
         r"Collapse types: \textit{bridge} = collapsed within 90-day window; "
         r"\textit{post\_bridge} = screen + $R$-run rule; "
         r"\textit{end\_of\_sample} = extends to window end.}"),
        r"\label{tab:svadf_equity}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        (r" & & & "
         r"\multicolumn{6}{c}{\textbf{W1 post-ChatGPT (" + W1_LABEL.replace("→", r"$\to$") + r")}} & "
         r"\multicolumn{6}{c}{\textbf{W2 pre-ChatGPT (" + W2_LABEL.replace("→", r"$\to$") + r")}} \\"),
        r"\cmidrule(lr){4-9} \cmidrule(lr){10-15}",
        (r"Segment & Ticker & Company & "
         r"$T$ & Det. & Start & End & Dur. & Collapse & "
         r"$T$ & Det. & Start & End & Dur. & Collapse \\"),
        r"\midrule",
    ]

    prev_seg = None
    for _, row in df.iterrows():
        seg = row["Segment"]
        if seg != prev_seg and prev_seg is not None:
            lines.append(r"\midrule")
        prev_seg = seg

        cells = [
            _latex_escape(row["Segment"]),
            row["Ticker"],
            _latex_escape(row["Company"]),
            str(row["W1 T"]),
            (r"\textcolor{teal}{\textbf{Yes}}"
             if row["W1 Detected"] == "Yes"
             else r"\textcolor{gray}{No}"),
            _latex_escape(row["W1 Start"]),
            _latex_escape(row["W1 End"]),
            _latex_escape(row["W1 Duration"]),
            r"\textit{" + _latex_escape(row["W1 Collapse"]) + "}",
            str(row["W2 T"]),
            (r"\textcolor{teal}{\textbf{Yes}}"
             if row["W2 Detected"] == "Yes"
             else r"\textcolor{gray}{No}"),
            _latex_escape(row["W2 Start"]),
            _latex_escape(row["W2 End"]),
            _latex_escape(row["W2 Duration"]),
            r"\textit{" + _latex_escape(row["W2 Collapse"]) + "}",
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings.TABLES_SVADF_DIR.mkdir(parents=True, exist_ok=True)

    df = build_svadf_equity_table()

    csv_path = settings.TABLES_SVADF_DIR / "svadf_equity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV   -> {csv_path}")

    tex_path = settings.TABLES_SVADF_DIR / "svadf_equity_results.tex"
    tex_path.write_text(_to_latex(df), encoding="utf-8")
    print(f"LaTeX -> {tex_path}")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
