"""Live market-data feed for the A+B+Diversifier Sleeves strategy.

Supports Alpaca Market Data API as the primary source with yfinance as fallback.
All data is cached on disk under ``cache/`` (repo-local, gitignored) so backfills
and restarts are fast and deterministic.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Optional Alpaca SDK.
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:  # pragma: no cover
    StockHistoricalDataClient = None  # type: ignore

# Optional yfinance fallback.
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent  # live/ -> repo root
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_alpaca_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read Alpaca API keys from environment variables."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    return key, secret


def _load_cache(ticker: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    path = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"
    df.to_parquet(path)


def _merge_with_cache(ticker: str, fresh: pd.DataFrame) -> pd.DataFrame:
    cached = _load_cache(ticker)
    if cached.empty:
        merged = fresh
    else:
        merged = pd.concat([cached, fresh], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    _save_cache(ticker, merged)
    return merged


def _fetch_alpaca(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> pd.DataFrame:
    if StockHistoricalDataClient is None:
        raise RuntimeError("alpaca-py is not installed")
    key, secret = _get_alpaca_credentials()
    api_key = api_key or key
    api_secret = api_secret or secret
    if not api_key or not api_secret:
        raise ValueError("Alpaca credentials missing")

    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
        adjustment="all",
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    if df.empty:
        raise ValueError(f"Alpaca returned no bars for {ticker}")

    df = df[df["symbol"] == ticker] if "symbol" in df.columns else df
    df = df.rename(columns={
        "timestamp": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vwap": "vwap",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    return df.loc[:end]


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    prefer_alpaca: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a single ticker, using cache and fallback logic.

    Parameters
    ----------
    ticker : str
        yfinance-compatible ticker (e.g. "SPY", "BTC-USD").
    start : date
        Inclusive start date.
    end : date
        Inclusive end date.
    prefer_alpaca : bool
        If True, try Alpaca first; otherwise yfinance first.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns open, high, low, close, volume.
    """
    # First, try to satisfy the request from cache if it already covers the range.
    cached = _load_cache(ticker)
    if not cached.empty and cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
        return cached.loc[start:end]

    errors: List[str] = []
    sources = [_fetch_alpaca, _fetch_yfinance] if prefer_alpaca else [_fetch_yfinance, _fetch_alpaca]

    for source in sources:
        try:
            if source is _fetch_alpaca and ("/" in ticker or "-" in ticker):
                # Alpaca stocks only; skip crypto-like yfinance tickers.
                continue
            fresh = source(ticker, start, end)
            if not fresh.empty:
                return _merge_with_cache(ticker, fresh).loc[start:end]
        except Exception as e:
            errors.append(f"{source.__name__}: {e}")

    raise RuntimeError(f"Could not fetch {ticker}: {'; '.join(errors)}")


def fetch_panel(
    tickers: List[str],
    start: date,
    end: date,
    prefer_alpaca: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for many tickers aligned to a common date index.

    Parameters
    ----------
    tickers : list[str]
    start : date
    end : date
    prefer_alpaca : bool

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = trading dates, values = adjusted close.
    """
    frames: Dict[str, pd.DataFrame] = {}
    failures: List[str] = []
    for t in tickers:
        try:
            frames[t] = fetch_ohlcv(t, start, end, prefer_alpaca=prefer_alpaca)
        except Exception as e:
            failures.append(f"{t}: {e}")

    if not frames:
        raise RuntimeError(f"No tickers could be downloaded: {'; '.join(failures)}")

    all_dates = sorted(set().union(*(df.index for df in frames.values())))
    aligned = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name="date"), columns=tickers, dtype=float)
    for t, df in frames.items():
        aligned[t] = df["close"].reindex(aligned.index)
    aligned = aligned.ffill(limit=1)
    return aligned


def get_last_trading_day(end: date = date.today()) -> date:
    """Return the most recent US-equity trading day on or before ``end``."""
    ts = pd.Timestamp(end)
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts.date()
