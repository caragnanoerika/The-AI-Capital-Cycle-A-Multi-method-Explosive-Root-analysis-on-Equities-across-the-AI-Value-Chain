"""
Structured results I/O.

Each method writes a JSON summary (small, human-readable, HF-friendly) and
optionally a parquet file for path/time-series data. Reading and writing
are pure functions — no global state. Easy to call from a Gradio app.

Layout:
    data/results/stationarity/<ticker>.json
    data/results/sadf/<ticker>.json
    data/results/gsadf/<ticker>_summary.json
    data/results/gsadf/<ticker>_paths.parquet
    data/results/svadf/<ticker>_<window_id>_summary.json
    data/results/svadf/<ticker>_<window_id>_paths.parquet
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from config import settings
from src.io.window import window_id


# ── JSON helper (handles dates, numpy types) ─────────────────────────────────
def _to_jsonable(obj):
    if isinstance(obj, (np.integer,)):      return int(obj)
    if isinstance(obj, (np.floating,)):     return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):         return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):    return obj.isoformat()
    if isinstance(obj, (date, datetime)):   return obj.isoformat()
    if isinstance(obj, dict):               return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):      return [_to_jsonable(v) for v in obj]
    return obj


def _dump_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Stationarity ─────────────────────────────────────────────────────────────
def save_stationarity(ticker: str, data: dict) -> Path:
    p = settings.RES_STATIONARITY_DIR / f"{ticker}.json"
    _dump_json(data, p)
    return p


def load_stationarity(ticker: str) -> dict | None:
    return _load_json(settings.RES_STATIONARITY_DIR / f"{ticker}.json")


# ── SADF ─────────────────────────────────────────────────────────────────────
def save_sadf(ticker: str, data: dict) -> Path:
    p = settings.RES_SADF_DIR / f"{ticker}.json"
    _dump_json(data, p)
    return p


def load_sadf(ticker: str) -> dict | None:
    return _load_json(settings.RES_SADF_DIR / f"{ticker}.json")


def save_sadf_paths(ticker: str, paths_df: pd.DataFrame) -> Path:
    p = settings.RES_SADF_DIR / f"{ticker}_paths.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    paths_df.to_parquet(p)
    return p


def load_sadf_paths(ticker: str) -> pd.DataFrame | None:
    p = settings.RES_SADF_DIR / f"{ticker}_paths.parquet"
    return pd.read_parquet(p) if p.exists() else None


def save_sadf_window(ticker: str, w_start: str, w_end: str,
                     summary: dict, paths_df: pd.DataFrame | None = None) -> dict:
    """Save SADF results for an arbitrary sub-window (mirrors SV-ADF pattern)."""
    wid = window_id(w_start, w_end)
    sp = settings.RES_SADF_DIR / f"{ticker}_{wid}_summary.json"
    _dump_json(summary, sp)
    pp = None
    if paths_df is not None:
        pp = settings.RES_SADF_DIR / f"{ticker}_{wid}_paths.parquet"
        pp.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_parquet(pp)
    return {"summary": sp, "paths": pp}


def load_sadf_window(ticker: str, w_start: str, w_end: str) -> dict | None:
    wid = window_id(w_start, w_end)
    summary = _load_json(settings.RES_SADF_DIR / f"{ticker}_{wid}_summary.json")
    if summary is None:
        return None
    pp = settings.RES_SADF_DIR / f"{ticker}_{wid}_paths.parquet"
    paths_df = pd.read_parquet(pp) if pp.exists() else None
    return {"summary": summary, "paths": paths_df}


# ── GSADF ────────────────────────────────────────────────────────────────────
def save_gsadf(ticker: str, summary: dict,
               paths_df: pd.DataFrame | None = None) -> dict:
    sp = settings.RES_GSADF_DIR / f"{ticker}_summary.json"
    _dump_json(summary, sp)
    pp = None
    if paths_df is not None:
        pp = settings.RES_GSADF_DIR / f"{ticker}_paths.parquet"
        pp.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_parquet(pp)
    return {"summary": sp, "paths": pp}


def load_gsadf(ticker: str) -> dict | None:
    summary = _load_json(settings.RES_GSADF_DIR / f"{ticker}_summary.json")
    if summary is None:
        return None
    pp = settings.RES_GSADF_DIR / f"{ticker}_paths.parquet"
    paths_df = pd.read_parquet(pp) if pp.exists() else None
    return {"summary": summary, "paths": paths_df}


# ── SV-ADF ───────────────────────────────────────────────────────────────────
def save_svadf(ticker: str, w_start: str, w_end: str,
               summary: dict, paths_df: pd.DataFrame | None = None) -> dict:
    wid = window_id(w_start, w_end)
    sp = settings.RES_SVADF_DIR / f"{ticker}_{wid}_summary.json"
    _dump_json(summary, sp)
    pp = None
    if paths_df is not None:
        pp = settings.RES_SVADF_DIR / f"{ticker}_{wid}_paths.parquet"
        pp.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_parquet(pp)
    return {"summary": sp, "paths": pp}


def load_svadf(ticker: str, w_start: str, w_end: str) -> dict | None:
    wid = window_id(w_start, w_end)
    summary = _load_json(settings.RES_SVADF_DIR / f"{ticker}_{wid}_summary.json")
    if summary is None:
        return None
    pp = settings.RES_SVADF_DIR / f"{ticker}_{wid}_paths.parquet"
    paths_df = pd.read_parquet(pp) if pp.exists() else None
    return {"summary": summary, "paths": paths_df}


def list_svadf_windows(ticker: str) -> list[tuple[str, str]]:
    """Return all (start, end) pairs for which SV-ADF results exist on disk."""
    out = []
    for f in settings.RES_SVADF_DIR.glob(f"{ticker}_*_summary.json"):
        wid = f.stem.replace(f"{ticker}_", "").replace("_summary", "")
        try:
            start, end = wid.split("_")
            out.append((start, end))
        except ValueError:
            continue
    return sorted(out)


# ── Discovery helpers (for an HF app) ────────────────────────────────────────
def available_tickers() -> list[str]:
    """Tickers with at least one result on disk."""
    tickers = set()
    for d in [settings.RES_STATIONARITY_DIR, settings.RES_SADF_DIR,
              settings.RES_GSADF_DIR, settings.RES_SVADF_DIR]:
        if d.exists():
            for f in d.glob("*.json"):
                tk = f.stem.replace("_summary", "").split("_")[0]
                tickers.add(tk)
    return sorted(tickers)


def available_methods(ticker: str) -> list[str]:
    """Which methods have results for this ticker."""
    out = []
    if (settings.RES_STATIONARITY_DIR / f"{ticker}.json").exists():
        out.append("stationarity")
    if (settings.RES_SADF_DIR / f"{ticker}.json").exists():
        out.append("sadf")
    if (settings.RES_GSADF_DIR / f"{ticker}_summary.json").exists():
        out.append("gsadf")
    if list_svadf_windows(ticker):
        out.append("svadf")
    return out
