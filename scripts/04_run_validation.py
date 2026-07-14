"""
Step 4 — Validation on historical bubbles.

Applies all methods to NASDAQ dot-com (1990-2003) and S&P 500 around the
dot-com peak and the GFC. Produces the same diagnostic figures as the
main analysis.

Usage:
    python scripts/04_run_validation.py
    python scripts/04_run_validation.py --case NASDAQ_dotcom
    python scripts/04_run_validation.py --methods gsadf svadf
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from config import settings
from src.methods.gsadf      import warmup
from src.pipelines.stationarity import run_one as run_stationarity
from src.pipelines.sadf         import run_one as run_sadf
from src.pipelines.gsadf        import run_one as run_gsadf
from src.pipelines.svadf        import run_one as run_svadf
from src.plotting.comparison    import plot_comparison


def fetch_case(case: dict) -> pd.Series | None:
    raw = yf.download(case["ticker"], start=case["start"], end=case["end"],
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return None
    s = raw["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index)
    s.name  = case["ticker"]
    return s


def main(cases: list[str] | None, methods: list[str]) -> None:
    if "all" in methods:
        methods = ["stationarity", "sadf", "gsadf", "svadf"]
    if "gsadf" in methods:
        warmup()
    cases = cases or list(settings.VALIDATION_CASES.keys())

    for name in cases:
        case = settings.VALIDATION_CASES[name]
        print(f"\n{'='*70}\n  {name}: {case['ticker']}  "
              f"({case['start']} → {case['end']})\n"
              f"  Expected: {case['expected_dates']}\n{'='*70}")
        series = fetch_case(case)
        if series is None:
            print("  download failed"); continue
        print(f"  T = {len(series)}")

        # Persist the validation series to prices.csv under the case name so
        # load_ticker_series can find it for plotting. The historical date
        # range doesn't interfere with the main 2019-2026 universe.
        if settings.PRICES_FILE.exists():
            prices_df = pd.read_csv(settings.PRICES_FILE, index_col=0, parse_dates=True)
        else:
            prices_df = pd.DataFrame()
        prices_df[name] = series
        prices_df.to_csv(settings.PRICES_FILE)

        # Treat the validation case as a "ticker" with its own name so results
        # write to standard folders.
        if "stationarity" in methods:
            run_stationarity(name, series, force=True)
        if "sadf" in methods:
            run_sadf(name, series, force=True)
        if "gsadf" in methods:
            run_gsadf(name, series, force=True, force_mc=False, verbose=True)
        if "svadf" in methods:
            run_svadf(name, series,
                      window_start=case["start"], window_end=case["end"],
                      force=True)

        # Save the comparison figure to outputs/figures
        fig_path = settings.FIGURES_DIR / f"validation_{name}.png"
        if "gsadf" in methods and "svadf" in methods:
            plot_comparison(name,
                            svadf_window=(case["start"], case["end"]),
                            save_path=fig_path, show=False)
            print(f"  saved figure → {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=None,
                   help="Subset of validation cases (default: all)")
    p.add_argument("--methods", nargs="+", default=["all"],
                   choices=["stationarity", "sadf", "gsadf", "svadf", "all"],
                   help="Which methods to run")
    args = p.parse_args()
    main(cases=args.cases, methods=args.methods)
