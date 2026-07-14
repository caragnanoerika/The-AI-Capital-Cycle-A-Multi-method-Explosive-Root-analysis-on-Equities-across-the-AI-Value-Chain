"""
Global configuration. The only file you should normally edit to change the
analysis: sample windows, test parameters, ticker universe, output paths.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
PRICES_DIR      = DATA_DIR / "prices"
MC_CACHE_DIR    = DATA_DIR / "mc_cache"
MC_PATHS_DIR    = MC_CACHE_DIR / "bsadf_paths"
RESULTS_DIR     = DATA_DIR / "results"
RES_STATIONARITY_DIR = RESULTS_DIR / "stationarity"
RES_SADF_DIR    = RESULTS_DIR / "sadf"
RES_GSADF_DIR   = RESULTS_DIR / "gsadf"
RES_SVADF_DIR   = RESULTS_DIR / "svadf"

OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
TABLES_DIR      = OUTPUTS_DIR / "tables"
FIGURES_DIR     = OUTPUTS_DIR / "figures"

# outputs/tables/ subfolders — one per producing method, mirrors outputs/figures/
TABLES_STATIONARITY_DIR = TABLES_DIR / "stationarity"  # scripts/06_stationarity_table.py
TABLES_GSADF_DIR        = TABLES_DIR / "gsadf"          # scripts/12_gsadf_equity_table.py
TABLES_SVADF_DIR        = TABLES_DIR / "svadf"          # scripts/13_svadf_equity_table.py
TABLES_SUMMARY_DIR      = TABLES_DIR / "summary"        # notebooks/01_visualization.ipynb

PRICES_FILE          = PRICES_DIR / "prices.csv"
MC_GLOBAL_INDEX_FILE = MC_CACHE_DIR / "global_critical_values.csv"

# ─── Sample windows ───────────────────────────────────────────────────────────
# Fixed window for stationarity tests, SADF, and GSADF.
# These are global properties of the series and don't change with window choice.
FIXED_START = "2020-01-01"
FIXED_END   = "2026-05-01"
FREQUENCY   = "1d"

# Default window for SV-ADF (override per run with --svadf-start / --svadf-end).
# Sarkar (2026, personal communication): use narrower windows for SV-ADF since
# its expanding recursion otherwise gets dominated by the earliest episode.
SVADF_DEFAULT_START = "2021-11-01"
SVADF_DEFAULT_END   = "2026-05-01"

# Pre-ChatGPT comparison window for SV-ADF.  Covers the portion of the fixed
# sample that SVADF_DEFAULT_START leaves out, enabling a direct pre- vs
# post-ChatGPT comparison on the same GSADF sample period.
SVADF_PRE_GPT_START = "2020-01-01"
SVADF_PRE_GPT_END   = "2022-10-31"

# ─── Test parameters ──────────────────────────────────────────────────────────
ADF_MAX_LAGS   = "AIC"
PP_LAGS        = "short"
KPSS_LAGS      = "auto"

GSADF_MC_REPS  = 999      # Monte Carlo reps for GSADF critical values
GSADF_QUANTILE = 0.95
SADF_CV_5PCT   = 1.49     # Asymptotic 5% CV (PWY 2011)
SEED           = 42

# SV-ADF episode-detection parameters (Sarkar & Wells 2026, reference code)
SV_MIN_UP    = 42  # M: ≥ 42 consecutive steps above log(τ)/10 for origination
SV_MIN_DOWN  = 21  # R: ≥ 21 consecutive steps below log(τ)/2  for post-bridge collapse
SV_BRIDGE_DAYS = 90  # bridge window in calendar days after origination

# GSADF/BSADF episode-detection filters — addresses spurious-bubble issue.
# A "minimum window" for an episode = how long a BSADF crossing must persist
# to count as a real episode. Defaults are MUCH stricter than before (was ~8).
EPISODE_MIN_DURATION = 60   # ≥ 60 trading days (~3 months) — filters out micro-episodes
EPISODE_MERGE_GAP    = 42   # merge episodes separated by ≤ 42 trading days

# ─── Universe ─────────────────────────────────────────────────────────────────
# Editing these dicts adds/removes tickers. New tickers trigger fresh Monte
# Carlo simulation for their (T, min_window) pair; existing ones reuse the cache.
# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSE — Equity universe organised by AI value-chain layer (Chapter 2).
# Ordering is strictly upstream → midstream → downstream, mirroring §2.2.
# Segments 1–5 are upstream (physical capacity), 6–8 are midstream (compute
# intermediation), 9–11 are downstream (applications & end users).
# ──────────────────────────────────────────────────────────────────────────────
UNIVERSE = {
    # ─── UPSTREAM (physical capacity) ────────────────────────────────────────
    "1. Semiconductor Equipment": {
        "ASML": "ASML — EUV lithography monopoly",
        "AMAT": "Applied Materials — wafer fab equipment",
        "LRCX": "Lam Research — etch/deposition equipment",
    },
    "2. Chip Design (AI accelerators)": {
        "NVDA": "Nvidia — GPU/AI accelerators, CUDA",
        "AMD":  "Advanced Micro Devices — Instinct/EPYC",
        "INTC": "Intel — CPUs, Gaudi accelerators",
        "AVGO": "Broadcom — networking + custom AI ASICs",
        "MRVL": "Marvell — custom silicon, AI networking",
    },
    "3. Memory (HBM-relevant)": {
        # Micron is the only US-listed HBM producer; the other two HBM firms
        # (SK Hynix, Samsung) are Korean-listed and outside the universe.
        "MU":   "Micron — DRAM/HBM",
    },
    "4. Foundry (leading-edge)": {
        # TSMC also represents the advanced-packaging layer (CoWoS) which
        # Chapter 2 §2.2 treats as a distinct bottleneck.
        "TSM":  "TSMC — leading-edge foundry + advanced packaging (CoWoS)",
    },
    "5. Storage (non-HBM control)": {
        # Included as a control / sentinel. Chapter 2 does not treat HDD/NAND
        # storage as part of the AI capital stack; explosivity here is read
        # against §5.4.3 as a within-segment negative reference for H2.
        "WDC":  "Western Digital — HDD/NAND storage (control)",
        "STX":  "Seagate — HDD storage (control)",
    },

    # ─── MIDSTREAM (compute intermediation) ──────────────────────────────────
    "6. Data-Centre Networking & Optical": {
        "ANET": "Arista Networks — DC switching",
        "CSCO": "Cisco — networking incumbent (dot-com analog)",
        "NOK":  "Nokia — telecom networks",
        "CIEN": "Ciena — optical networking",
    },
    "7. Data-Centre REITs (colocation & build-out)": {
        "EQIX": "Equinix — colocation",
        "DLR":  "Digital Realty — data-centre REIT",
    },
    "8. Hyperscale Cloud": {
        "MSFT":  "Microsoft — Azure, OpenAI partner",
        "AMZN":  "Amazon — AWS, Anthropic partner",
        "GOOGL": "Alphabet — Google Cloud, Gemini",
        "META":  "Meta — Llama, AI advertising",
        "ORCL":  "Oracle — OCI, AI infrastructure",
    },

    # ─── DOWNSTREAM (applications & end users) ───────────────────────────────
    "9. Enterprise AI / Hybrid Cloud": {
        "IBM":  "IBM — enterprise AI, hybrid cloud",
    },
    "10. Application & AI Platform Software": {
        "PLTR": "Palantir — enterprise/government AI",
        "SNOW": "Snowflake — data cloud",
        "DDOG": "Datadog — observability with AI",
        "AI":   "C3.ai — enterprise AI platform",
    },
    "11. Edge AI & Devices": {
        # NB: Chapter §3.1 formally excludes robotics/autonomous systems from
        # the AI-industry scope. TSLA is retained as an AI-narrative sentinel
        # and is read against the rest of the layer in §5.4.3.
        "QCOM": "Qualcomm — mobile/edge AI silicon",
        "AAPL": "Apple — devices, on-device AI",
        "TSLA": "Tesla — autonomy/robotics narrative (out-of-scope sentinel)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# PLANNED_ADDITIONS — Chapter 2 layers not yet represented in the universe.
# Identified in §5.2.1 as priority extensions; absence noted in §5.5.
# Promote to UNIVERSE once sample length is sufficient.
# ──────────────────────────────────────────────────────────────────────────────
PLANNED_ADDITIONS = {
    "Specialised AI Cloud (neoclouds)": {
        "CRWV": "CoreWeave — IPO Mar 2025; sample too short for first run",
        "NBIS": "Nebius — recently re-listed; sample too short for first run",
    },
    "Power & Grid (data-centre electricity)": {
        "CEG":  "Constellation Energy — nuclear, AI tie-ups (Microsoft)",
        "VST":  "Vistra — natural gas + nuclear, AI capex partners",
    },
    "Advanced Packaging (pure-play)": {
        # TSM already proxies for advanced packaging via CoWoS, but a pure-play
        # would tighten the §2.2 bottleneck test.
        "ASX":  "ASE Technology Holding — outsourced semiconductor packaging",
    },
    # Foundation models (OpenAI, Anthropic, xAI, Mistral, DeepSeek) are
    # privately held and lack continuous price series. Excluded by construction
    # rather than by choice; documented as an inherent limitation in §5.2.1.
}


# ──────────────────────────────────────────────────────────────────────────────
# COMMODITIES — Grouped by relevance to hypothesis H6 (propagation of
# explosive dynamics from AI equities into real-resource markets).
# AI-relevant groups carry the test; placebos serve as controls; precious
# metals are AI-correlated but carry a monetary confound and are read
# separately in §5.4.5.
# ──────────────────────────────────────────────────────────────────────────────
COMMODITIES = {
    # ─── AI-relevant: data-centre power and hardware-manufacturing inputs ──
    "AI-relevant: Energy (DC power proxy)": {
        "USO":  "United States Oil Fund ETF (WTI proxy)",
        "UNG":  "United States Natural Gas Fund ETF",
        "BNO":  "Brent Oil Fund ETF",
    },
    "AI-relevant: Industrial metals (hardware inputs)": {
        # NB: JJC ceased active trading mid-2023; replacement (CPER, COPX,
        # or JJN) is identified in §5.4.7 as outstanding work.
        "JJC":  "iShares Copper ETN — broken series since mid-2023",
        "PPLT": "Aberdeen Platinum ETF",
        "PALL": "Aberdeen Palladium ETF",
    },
    # ─── Precious metals: AI-correlated but monetary-driven (confound) ─────
    "Confound: Precious metals (monetary)": {
        "GLD":  "SPDR Gold ETF — flight-to-safety dynamics",
        "SLV":  "iShares Silver ETF — industrial + monetary use",
    },
    # ─── Placebos: no plausible AI linkage ─────────────────────────────────
    "Placebo: Agriculture": {
        "DBA":  "Invesco Agriculture ETF",
        "WEAT": "Teucrium Wheat ETF",
        "CORN": "Teucrium Corn ETF",
        "SOYB": "Teucrium Soybean ETF",
    },
    "Placebo: Livestock": {
        "COW":  "iPath Livestock ETN",
    },
    # ─── Aggregate references ──────────────────────────────────────────────
    "Benchmark: Broad commodity indices": {
        "DBC":  "Invesco DB Commodity Index ETF",
        "GSG":  "iShares S&P GSCI Commodity ETF",
        "PDBC": "Invesco Optimum Yield Diversified Commodity ETF",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# INDICES — Aggregate benchmarks for the H3 dilution test
# ──────────────────────────────────────────────────────────────────────────────
INDICES = {
    "^IXIC": "Nasdaq Composite — broad tech benchmark",
    "^SOX":  "PHLX Semiconductor Index — sector benchmark",
    "^GSPC": "S&P 500 — broad-market control",
}


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION_CASES — Historical bubble episodes for the H5 test (§5.3.8)
# ──────────────────────────────────────────────────────────────────────────────
VALIDATION_CASES = {
    # Matches Sarkar & Wells (2026) Figure 9 exactly: 1982-2012, M=360, R=360, bridge=720.
    # Expected SV-ADF dates: origination Apr 1995, collapse Sep 2000.
    "NASDAQ_dotcom": {
        "ticker": "^IXIC", "start": "1982-01-01", "end": "2012-12-31",
        "expected_peak": "March 2000",
        "expected_dates": "SV-ADF (Sarkar 2026): origination Apr 1995, collapse Sep 2000",
        "svadf_M": 360, "svadf_R": 360, "svadf_bridge": 720,
    },
    # Alternative configuration: shorter window (1990-2003) with proportionally
    # scaled parameters — used to illustrate parameter sensitivity in Chapter 5.
    "NASDAQ_dotcom_alt": {
        "ticker": "^IXIC", "start": "1990-01-01", "end": "2003-12-31",
        "expected_peak": "March 2000",
        "expected_dates": "SV-ADF (Sarkar 2026): origination Apr 1995, collapse Sep 2000",
        # No svadf_M/R/bridge → _scale_params uses proportional scaling from T
    },
}

def build_segment_lookup():
    """Build the master ticker → (segment, description) lookup.

    Iterates through UNIVERSE, PLANNED_ADDITIONS, COMMODITIES, INDICES and
    returns the canonical analysis ordering plus a ticker → (segment_label,
    description) mapping.

    Segment-label convention (the stable contract used downstream by
    comparison_table.csv, segment_summary.csv, and the plotting code):
      - Equity segments    : "1." through "11."   (upstream → downstream,
                                                   mirrors Chapter 2 §2.2)
      - Planned additions  : "P. <group>"          (Chapter 2 layers
                                                    promoted into the run)
      - Commodity segments : "C. <group>"
      - Index benchmarks   : "0. Index / Control"

    Downstream filters can therefore identify panels by prefix:
    Segment.startswith("C.") marks a commodity, "0." an index, "P." a
    planned addition, and any other leading-digit prefix the active
    equity universe.

    Returns
    -------
    all_tickers : list[str]
    segment_of  : dict[str, tuple[str, str]]
    """
    all_tickers, segment_of = [], {}

    # ── Equities (segments 1–11) ─────────────────────────────────────────────
    for segment, members in UNIVERSE.items():
        for tk, desc in members.items():
            all_tickers.append(tk)
            segment_of[tk] = (segment, desc)

    # ── Planned additions (Chapter 2 layers being promoted into the run) ────
    for segment, members in PLANNED_ADDITIONS.items():
        for tk, desc in members.items():
            all_tickers.append(tk)
            segment_of[tk] = (f"P. {segment}", desc)

    # ── Commodities (group name carries its H6 role: AI-relevant/Placebo/…) ─
    for segment, members in COMMODITIES.items():
        for tk, desc in members.items():
            all_tickers.append(tk)
            segment_of[tk] = (f"C. {segment}", desc)

    # ── Index benchmarks (H3 controls) ──────────────────────────────────────
    for tk, desc in INDICES.items():
        all_tickers.append(tk)
        segment_of[tk] = ("0. Index / Control", desc)

    # Defensive: a ticker should appear in exactly one panel.
    if len(all_tickers) != len(set(all_tickers)):
        from collections import Counter
        dup = [t for t, c in Counter(all_tickers).items() if c > 1]
        raise ValueError(f"Duplicate ticker(s) across panels: {dup}")

    return all_tickers, segment_of


ALL_TICKERS, SEGMENT_OF = build_segment_lookup()
