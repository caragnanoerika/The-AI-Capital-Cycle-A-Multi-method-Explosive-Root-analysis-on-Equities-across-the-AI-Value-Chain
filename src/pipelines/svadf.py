"""
Run SV-ADF for one ticker on a user-specified window.

The window can differ between runs. Each run produces its own result file
keyed by (ticker, window_id), so multiple windows can coexist on disk.
"""
from __future__ import annotations
import pandas as pd
from config import settings
from src.methods.svadf    import svadf
from src.methods.episodes import detect_sv_episode
from src.io.results       import save_svadf, load_svadf


def run_one(ticker: str, series: pd.Series,
            window_start: str | None = None,
            window_end:   str | None = None,
            force: bool = False,
            M:           int | None = None,
            R:           int | None = None,
            bridge_days: int | None = None) -> dict | None:
    """Run SV-ADF for `ticker` on [window_start, window_end] (defaults to the
    post-ChatGPT W1 window) and cache the result. Returns the summary dict,
    or None if skipped (cached) or the series is too short."""
    w_start     = window_start  or settings.SVADF_DEFAULT_START
    w_end       = window_end    or settings.SVADF_DEFAULT_END
    M           = M           or settings.SV_MIN_UP
    R           = R           or settings.SV_MIN_DOWN
    bridge_days = bridge_days or settings.SV_BRIDGE_DAYS

    if not force and load_svadf(ticker, w_start, w_end) is not None:
        return None

    s = series.dropna().astype(float).loc[w_start:w_end]
    if len(s) < 100:
        return None
    y, T = s.to_numpy(), len(s)
    res  = svadf(y)
    if res is None:
        return None

    ep = detect_sv_episode(
        res["coef_stat"],
        res["orig_thr"],
        res["screen_thr"],
        res["coll_thr"],
        dates=s.index,
        M=M,
        R=R,
        bridge_days=bridge_days,
    )

    summary = {
        "ticker":       ticker,
        "window_id":    f"{w_start}_{w_end}",
        "T":            int(T),
        "start_date":   s.index[0].isoformat()[:10],
        "end_date":     s.index[-1].isoformat()[:10],
        "min_window":   int(res["min_window"]),
        "M":            int(M),
        "R":            int(R),
        "bridge_days":  int(bridge_days),
        "episode":      ep,   # may be None
    }
    paths_df = pd.DataFrame({
        "coef_stat":  res["coef_stat"],
        "orig_thr":   res["orig_thr"],
        "screen_thr": res["screen_thr"],
        "coll_thr":   res["coll_thr"],
    }, index=s.index)
    paths_df.index.name = "date"
    save_svadf(ticker, w_start, w_end, summary, paths_df)
    return summary
