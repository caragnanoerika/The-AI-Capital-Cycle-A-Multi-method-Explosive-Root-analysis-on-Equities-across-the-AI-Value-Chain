"""
Step 10 — SV-ADF sector-panel summary figures.

Generates one figure per segment showing all tickers' SV-ADF coefficient-
stat paths overlaid on the same time axis.  Both W1 (post-ChatGPT) and W2
(pre-ChatGPT) windows are shown on each row.

Output: outputs/figures/svadf_sector/<segment_slug>.png

Usage
-----
    python scripts/10_svadf_sector_panels.py          # all segments
    python scripts/10_svadf_sector_panels.py --segments "1. Semiconductor Equipment"
    python scripts/10_svadf_sector_panels.py --show    # display interactively
"""
from __future__ import annotations
import sys
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from config import settings
from src.plotting.diagnostic import plot_svadf_segment_panel

FIGURES_DIR = settings.FIGURES_DIR / "svadf_sector"


def _slug(segment: str) -> str:
    """Convert segment label to a safe filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", segment.lower()).strip("_")


def main(segments=None, show: bool = False) -> None:
    all_segs = sorted({seg for _, (seg, _) in settings.SEGMENT_OF.items()})

    if segments:
        # Accept exact match or partial substring
        target = []
        for s in segments:
            matches = [g for g in all_segs if s.lower() in g.lower()]
            if matches:
                target.extend(matches)
            else:
                print(f"  [WARN] segment '{s}' not found — skipping")
        all_segs = sorted(set(target))

    if not all_segs:
        print("No segments to plot."); return

    windows = [
        (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
        (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating SV-ADF sector panels → {FIGURES_DIR}")
    print(f"Windows: {windows}\n")

    for seg in all_segs:
        slug = _slug(seg)
        save_path = FIGURES_DIR / f"{slug}.png"
        print(f"  [{seg}]  →  {save_path.name}")
        fig = plot_svadf_segment_panel(
            segment=seg,
            windows=windows,
            save_path=save_path,
            show=show,
        )
        if fig is None:
            print(f"    [skipped — no results on disk]")

    print(f"\nDone.  Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", nargs="*", default=None,
                   help="Segment labels to include (default: all). "
                        "Partial match supported.")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively instead of saving")
    args = p.parse_args()
    main(segments=args.segments, show=args.show)
