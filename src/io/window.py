"""
Helpers for SV-ADF / windowed-SADF window identification.

Every result file for a date-window-scoped run (SV-ADF, or SADF re-run on a
custom sub-window) is tagged with a `{start}_{end}` window id so that results
from different windows can coexist on disk for the same ticker without
overwriting each other (see src/io/results.py).
"""
from __future__ import annotations


def window_id(start: str, end: str) -> str:
    """Construct a filename-safe window identifier, e.g. '2021-11-01_2026-05-01'."""
    return f"{start}_{end}"
