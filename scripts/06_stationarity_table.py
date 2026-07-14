"""
06_stationarity_table.py
Build a stationarity summary table from per-ticker JSON results.

Default mode  (no args)
  Reads cached JSON results (data/results/stationarity/<ticker>.json) and
  writes stationarity_table.csv.

Date-range mode  (--start / --end)
  Loads raw prices from the project cache, slices to the requested window,
  re-runs ADF / PP / KPSS fresh, and writes stationarity_table_<start>_<end>.csv.
  Use this for robustness checks across different sub-periods.

  Example:
    python scripts/06_stationarity_table.py --start 2022-11-01 --end 2026-05-01
    python scripts/06_stationarity_table.py --start 2020-01-01 --end 2022-10-31

Feeds into: scripts/07_stationarity_figure.py (reads this table's source JSON,
not the CSV itself, but must run after this step conceptually).

Output: outputs/tables/stationarity/stationarity_table[_<start>_<end>].csv
"""
import argparse
import json
import sys
from pathlib import Path

# ── project root resolution ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import settings

STAT_DIR = settings.RES_STATIONARITY_DIR


def _fmt_stat(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _decision(reject: bool) -> str:
    return "Reject" if reject else "Fail to reject"


def _implied_integration(adf_reject: bool, pp_reject: bool, kpss_reject: bool) -> str:
    """
    ADF / PP H0 = unit root  → reject means stationary (I(0) signal)
    KPSS      H0 = stationary → reject means non-stationary (I(1) signal)
    """
    nonstat_votes = (not adf_reject) + (not pp_reject) + kpss_reject
    stat_votes    = adf_reject + pp_reject + (not kpss_reject)

    if nonstat_votes == 3:
        return "I(1)"
    if stat_votes == 3:
        return "I(0)"
    if nonstat_votes >= 2:
        return "I(1)*"   # majority non-stationary, some ambiguity
    return "I(0)*"       # majority stationary, some ambiguity


def build_table_from_cache() -> pd.DataFrame:
    """Default mode: read pre-computed JSON results."""
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
        rows.append(_make_row(tk, seg, d["T"] if "T" in d else "",
                              d["adf"], d["pp"], d["kpss"]))
    return pd.DataFrame(rows)


def build_table_from_prices(start: str, end: str) -> pd.DataFrame:
    """Date-range mode: slice cached prices and re-run tests fresh."""
    from src.io.data import load_or_download_prices
    from src.methods.stationarity import run_adf, run_pp, run_kpss

    print(f"Loading prices and slicing {start} to {end} ...")
    prices = load_or_download_prices(force=False)
    prices = prices.loc[start:end]
    print(f"  Window: {prices.index[0].date()} to {prices.index[-1].date()}  "
          f"({len(prices)} trading days)")

    rows = []
    for tk in settings.ALL_TICKERS:
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        if len(s) < 30:
            print(f"  {tk}: skipped (T={len(s)} < 30)")
            continue
        seg, _ = settings.SEGMENT_OF.get(tk, ("", ""))
        adf  = run_adf(s,  max_lags=settings.ADF_MAX_LAGS)
        pp   = run_pp(s,   lags=settings.PP_LAGS)
        kpss = run_kpss(s, lags=settings.KPSS_LAGS)
        if adf and pp and kpss:
            rows.append(_make_row(tk, seg, len(s), adf, pp, kpss))
    return pd.DataFrame(rows)


def _make_row(tk, seg, T, adf, pp, kpss) -> dict:
    return {
        "Ticker"        : tk,
        "Segment"       : seg,
        "T"             : T,
        "ADF stat"      : _fmt_stat(adf["stat"]),
        "ADF decision"  : _decision(adf["reject_5pct"]),
        "PP stat"       : _fmt_stat(pp["stat"]),
        "PP decision"   : _decision(pp["reject_5pct"]),
        "KPSS stat"     : _fmt_stat(kpss["stat"], decimals=3),
        "KPSS decision" : _decision(kpss["reject_5pct"]),
        "Implied I(d)"  : _implied_integration(
                              adf["reject_5pct"], pp["reject_5pct"], kpss["reject_5pct"]),
    }


def _print_summary(df: pd.DataFrame, out_file: Path) -> None:
    total = len(df)
    i1    = df["Implied I(d)"].str.startswith("I(1)").sum()
    i0    = df["Implied I(d)"].str.startswith("I(0)").sum()
    print(f"Written: {out_file}")
    print(f"  {total} tickers  |  I(1): {i1}  |  I(0): {i0}  |  ambiguous: {total-i1-i0}")
    print()
    print(df[["Ticker", "ADF decision", "PP decision",
              "KPSS decision", "Implied I(d)"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None,
                        help="Window start date YYYY-MM-DD (triggers fresh re-run)")
    parser.add_argument("--end",   default=None,
                        help="Window end   date YYYY-MM-DD (triggers fresh re-run)")
    args = parser.parse_args()

    settings.TABLES_STATIONARITY_DIR.mkdir(parents=True, exist_ok=True)

    if args.start or args.end:
        start = args.start or settings.FIXED_START
        end   = args.end   or settings.FIXED_END
        df    = build_table_from_prices(start, end)
        tag   = f"_{start}_{end}"
        out   = settings.TABLES_STATIONARITY_DIR / f"stationarity_table{tag}.csv"
    else:
        df  = build_table_from_cache()
        out = settings.TABLES_STATIONARITY_DIR / "stationarity_table.csv"

    if df.empty:
        print("No results — check that prices are cached and stationarity JSONs exist.")
        sys.exit(1)

    df.to_csv(out, index=False, encoding="utf-8-sig")
    _print_summary(df, out)
