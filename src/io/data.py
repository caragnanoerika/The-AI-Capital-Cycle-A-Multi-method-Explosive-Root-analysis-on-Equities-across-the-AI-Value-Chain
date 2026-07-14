"""Price-data download and caching. Yahoo Finance via yfinance."""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from config import settings


def download_prices(tickers, start, end, interval="1d") -> pd.DataFrame:
    """Fetch adjusted close prices for `tickers` from Yahoo Finance and
    return a wide DataFrame indexed by date, one column per ticker."""
    print(f"Downloading {len(tickers)} tickers …")
    raw = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({
            tk: raw[tk]["Close"] for tk in tickers
            if tk in raw.columns.get_level_values(0)
        })
    else:
        prices = pd.DataFrame({tickers[0]: raw["Close"]})
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def load_or_download_prices(force: bool = False) -> pd.DataFrame:
    """Return wide DataFrame date × ticker for the FIXED sample window."""
    path = settings.PRICES_FILE
    if path.exists() and not force:
        print(f"Loading cached prices from {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    settings.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    prices = download_prices(
        settings.ALL_TICKERS, start=settings.FIXED_START,
        end=settings.FIXED_END, interval=settings.FREQUENCY,
    ).ffill()
    prices.to_csv(path)
    print(f"Saved prices to {path}")
    return prices


def load_ticker_series(ticker: str, start: str | None = None,
                       end: str | None = None) -> pd.Series:
    """
    Return the price series for one ticker, optionally sliced to a window.
    Falls back to downloading just that ticker if not in the main file.
    """
    if settings.PRICES_FILE.exists():
        prices = pd.read_csv(settings.PRICES_FILE, index_col=0, parse_dates=True)
        if ticker in prices.columns:
            s = prices[ticker].dropna()
            if start: s = s.loc[start:]
            if end:   s = s.loc[:end]
            return s.dropna()
    # Fall back to single-ticker download
    df = download_prices([ticker],
                         start=start or settings.FIXED_START,
                         end=end or settings.FIXED_END)
    return df[ticker].dropna()
