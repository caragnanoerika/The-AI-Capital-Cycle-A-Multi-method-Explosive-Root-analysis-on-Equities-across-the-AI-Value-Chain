"""
Step 11 — Per-ticker SV-ADF volatility analysis.

For every ticker in the universe, generates a four-panel figure:

  1. Adjusted close price + detected episode shading
  2. Rolling annualised volatility (default: 40-day ≈ 8 weeks)
     with high-volatility regime shading (> 75th percentile)
  3. Recursive SV-ADF statistic with all three thresholds
     GREY shading = near-miss crossings (stat > orig_thr for < M
     consecutive steps) — these are regime shifts that SV-ADF correctly
     avoids classifying as bubbles
  4. Timeline strip: detected episode / near-miss windows / high-vol regimes

The contrast between the grey near-miss bands, the blue high-vol shading,
and the orange episode shading makes it straightforward to infer whether
threshold crossings are driven by genuine explosiveness or transient
volatility spikes.

Output: outputs/figures/svadf_volatility/<TICKER>.png

Usage
-----
    python scripts/11_volatility_analysis.py               # all tickers
    python scripts/11_volatility_analysis.py --tickers NVDA AMD GOOGL
    python scripts/11_volatility_analysis.py --window W2   # pre-ChatGPT
    python scripts/11_volatility_analysis.py --vol-window 20  # 4-week vol
"""
from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from config import settings
from src.io.results import list_svadf_windows
from src.plotting.diagnostic import plot_svadf_volatility

FIGURES_DIR = settings.FIGURES_DIR / "svadf_volatility"


def _resolve_window(w_arg: str | None) -> tuple[str, str] | None:
    """Map 'W1'/'W2' shortcuts or None to (start, end) tuple."""
    if w_arg is None or w_arg.upper() == "W1":
        return (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
    if w_arg.upper() == "W2":
        return (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END)
    # Treat as literal "start_end" or "start,end"
    parts = w_arg.replace(",", "_").split("_", maxsplit=1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


def main(tickers=None, window_arg=None, vol_window: int = 40,
         show: bool = False) -> None:
    universe = tickers or settings.ALL_TICKERS
    window   = _resolve_window(window_arg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating SV-ADF volatility figures → {FIGURES_DIR}")
    print(f"Window: {window}  |  vol_window: {vol_window}d  |  tickers: {len(universe)}\n")

    t0 = time.time()
    done = skip = fail = 0

    for tk in universe:
        # Verify at least one SV-ADF result exists
        avail = list_svadf_windows(tk)
        if not avail:
            print(f"  [{tk}] no SV-ADF results — skipping"); skip += 1; continue

        save_path = FIGURES_DIR / f"{tk}.png"
        try:
            fig = plot_svadf_volatility(
                ticker=tk,
                window=window,
                vol_window=vol_window,
                save_path=save_path,
                show=show,
            )
            if fig is not None:
                print(f"  [{tk}] saved → {save_path.name}")
                done += 1
            else:
                print(f"  [{tk}] no data for window {window}"); skip += 1
        except Exception as exc:
            print(f"  [{tk}] ERROR: {exc}"); fail += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  saved={done}  skipped={skip}  failed={fail}")
    if fail:
        print("  Check error messages above for failed tickers.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers (default: all from settings)")
    p.add_argument("--window", default=None,
                   help="SV-ADF window: 'W1' (default/post-ChatGPT), "
                        "'W2' (pre-ChatGPT), or 'YYYY-MM-DD_YYYY-MM-DD'")
    p.add_argument("--vol-window", type=int, default=40,
                   help="Rolling volatility window in trading days (default: 40 ≈ 8 weeks)")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively (not recommended for large universe)")
    args = p.parse_args()
    main(tickers=args.tickers, window_arg=args.window,
         vol_window=args.vol_window, show=args.show)
