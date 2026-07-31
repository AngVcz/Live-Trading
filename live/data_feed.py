"""Live market-data feed for the A+B+Diversifier Sleeves strategy.

Supports Alpaca Market Data API as the primary source with yfinance as fallback.
All data is cached on disk under ``cache/`` (repo-local, gitignored) so backfills
and restarts are fast and deterministic.

Calendar/provenance notes:
- The panel is built on the INTERSECTION of equity (non-crypto) frames' indexes,
  so weekend crypto bars never leak into the equity calendar.
- Symbols not tradeable on Alpaca (^VIX, crypto/futures pairs) skip Alpaca and go
  straight to yfinance; the source that produced each series is recorded in
  ``panel.attrs["provenance"]`` (ticker -> "alpaca"|"yfinance").
- Cache files are keyed per source (``{ticker}__{source}.parquet``) so a source
  switch can't stitch mixed adjusted-close conventions. A live-relevant cache hit
  is re-validated against a staleness window and its tail refreshed when stale.
- ``get_last_trading_day`` skips NYSE holidays (reuses the holiday set from
  ``live.risk``) so weekend + holiday runs resolve to the prior trading day.
"""
from __future__ import annotations

import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# NYSE holiday set (no circular import: live.risk imports neither data_feed nor live).
from live.risk import _market_holidays

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

# Cache tail must be within this many days of "now" for a live-relevant hit to be
# considered fresh; otherwise the tail is refreshed from the live source.
STALENESS_DAYS = 1
# Only re-validate the tail when the request end is within this many days of today
# (keeps historical/backtest requests from paying a tail-refresh fetch).
LIVE_RELEVANT_DAYS = 7

# ponytail: explicit allowlist of known non-equity symbols that Alpaca doesn't carry.
# `^`-prefixed indices and `/`/`-` crypto-futures pairs are skipped via the prefix/rule
# below; this set is kept for clarity / future non-prefixed index tickers.
_NON_ALPACA_TICKERS: set[str] = {"^VIX"}


def _today() -> date:
    """Indirection so tests can monkeypatch the 'now' used by staleness checks."""
    return date.today()


def _get_alpaca_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read Alpaca API keys from environment variables."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    return key, secret


# --- source classification / skip logic (#15) ------------------------------
def _skip_alpaca(ticker: str) -> bool:
    """True if ``ticker`` is NOT tradeable on Alpaca (go straight to yfinance)."""
    return (
        ticker.startswith("^")
        or "/" in ticker
        or "-" in ticker
        or ticker in _NON_ALPACA_TICKERS
    )


def _is_crypto(ticker: str) -> bool:
    """True for crypto/futures pairs (trade 7 days/week, carry weekend bars)."""
    return "/" in ticker or "-" in ticker


def _ordered_sources(ticker: str, prefer_alpaca: bool) -> List[Tuple[str, Callable]]:
    """Ordered (name, fetch_fn) list; Alpaca is skipped for non-tradeable symbols."""
    if _skip_alpaca(ticker):
        return [("yfinance", _fetch_yfinance)]
    if prefer_alpaca:
        return [("alpaca", _fetch_alpaca), ("yfinance", _fetch_yfinance)]
    return [("yfinance", _fetch_yfinance), ("alpaca", _fetch_alpaca)]


# --- per-source disk cache (#16) -------------------------------------------
def _cache_path(ticker: str, source: str) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}__{source}.parquet"


def _load_cache(ticker: str, source: str) -> pd.DataFrame:
    path = _cache_path(ticker, source)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save_cache(ticker: str, source: str, df: pd.DataFrame) -> None:
    path = _cache_path(ticker, source)
    df.to_parquet(path)


def _merge_with_cache(ticker: str, source: str, fresh: pd.DataFrame) -> pd.DataFrame:
    cached = _load_cache(ticker, source)
    if cached.empty:
        merged = fresh
    else:
        merged = pd.concat([cached, fresh], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    _save_cache(ticker, source, merged)
    return merged


# --- staleness helpers (#16 stale-cache-hit) --------------------------------
def _is_live_relevant(end: date, now: Optional[date] = None) -> bool:
    """True when the request end is near today (a live run, not a backtest)."""
    now = now or _today()
    return (now - end).days <= LIVE_RELEVANT_DAYS


def _cache_is_stale(cached: pd.DataFrame, staleness_days: int = STALENESS_DAYS,
                    now: Optional[date] = None) -> bool:
    """True if the cache's last bar is more than ``staleness_days`` behind now."""
    if cached.empty:
        return True
    now = now or _today()
    last = cached.index.max()
    if isinstance(last, pd.Timestamp):
        last = last.date()
    return (now - last).days > staleness_days


# --- fetchers --------------------------------------------------------------
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
    staleness_days: int = STALENESS_DAYS,
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
        If True, try Alpaca first; otherwise yfinance first. Symbols not
        tradeable on Alpaca always go straight to yfinance.
    staleness_days : int
        For a live-relevant cache hit, refresh the tail if the cache's last bar
        is more than this many days behind today (default 1).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns open, high, low, close, volume.
        ``df.attrs["source"]`` records which source produced the series.
    """
    sources = _ordered_sources(ticker, prefer_alpaca)
    errors: List[str] = []

    for sname, sfn in sources:
        cached = _load_cache(ticker, sname)
        # Full coverage hit: cache already spans the requested range.
        if (
            not cached.empty
            and cached.index.min() <= pd.Timestamp(start)
            and cached.index.max() >= pd.Timestamp(end)
        ):
            if _is_live_relevant(end) and _cache_is_stale(cached, staleness_days):
                # Refresh the tail up to today so live runs pull newly-available bars.
                try:
                    fetch_start = (cached.index.max() + timedelta(days=1)).date()
                    if fetch_start <= _today():
                        fresh = sfn(ticker, fetch_start, _today())
                        cached = _merge_with_cache(ticker, sname, fresh)
                except Exception as e:
                    # ponytail: tail-refresh is best-effort — a covering cache must
                    # still be served even if the refresh hits a network blip, so we
                    # log the error and fall through to the return below (no `continue`).
                    errors.append(f"{sname} tail-refresh: {e}")
            result = cached.loc[start:end].copy()
            result.attrs["source"] = sname
            return result
        # Need to fetch: gap between cache and requested end (or no cache).
        try:
            fetch_start = start
            if not cached.empty:
                fetch_start = (cached.index.max() + timedelta(days=1)).date()
            fresh = sfn(ticker, fetch_start, end)
            if not fresh.empty:
                merged = _merge_with_cache(ticker, sname, fresh)
                result = merged.loc[start:end].copy()
                result.attrs["source"] = sname
                return result
        except Exception as e:
            errors.append(f"{sname}: {e}")

    raise RuntimeError(f"Could not fetch {ticker}: {'; '.join(errors)}")


def fetch_panel(
    tickers: List[str],
    start: date,
    end: date,
    prefer_alpaca: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for many tickers aligned to an EQUITY trading
    calendar (the intersection of all non-crypto frames' indexes).

    Crypto frames are reindexed onto that calendar with ``ffill(limit=1)`` so a
    weekend crypto close is carried forward at most one day; weekdays with no
    recent crypto bar remain NaN (not fabricated). The panel has NO weekend rows.

    ``panel.attrs["provenance"]`` maps each ticker to the source that produced it
    (``"alpaca"`` or ``"yfinance"``).
    """
    frames: Dict[str, pd.DataFrame] = {}
    provenance: Dict[str, str] = {}
    failures: List[str] = []
    for t in tickers:
        try:
            frames[t] = fetch_ohlcv(t, start, end, prefer_alpaca=prefer_alpaca)
            provenance[t] = frames[t].attrs.get("source", "unknown")
        except Exception as e:
            failures.append(f"{t}: {e}")

    if not frames:
        raise RuntimeError(f"No tickers could be downloaded: {'; '.join(failures)}")

    equity_frames = {t: df for t, df in frames.items() if not _is_crypto(t)}
    crypto_frames = {t: df for t, df in frames.items() if _is_crypto(t)}

    if not equity_frames:
        raise RuntimeError(
            "No equity (non-crypto) frames produced; cannot build a calendar. "
            f"Refusing to return a crypto-only panel. Failures: {'; '.join(failures)}"
        )
    if failures:
        # Surface a silent partial equity failure so it is not buried in the all-fail path.
        warnings.warn(
            f"fetch_panel: partial fetch failure ({len(failures)} ticker(s)): "
            f"{'; '.join(failures)} — affected columns will be all-NaN.",
            stacklevel=2,
        )

    # Intersection of equity frames' indexes = the trading calendar.
    # NOTE: the intersection truncates history to the shortest-lived equity frame,
    # so a late-listed or delisted ticker shrinks the backtest span.
    cal = None
    for df in equity_frames.values():
        cal = df.index if cal is None else cal.intersection(df.index)
    cal = cal.sort_values()

    aligned = pd.DataFrame(index=pd.DatetimeIndex(cal, name="date"), columns=tickers, dtype=float)
    for t, df in equity_frames.items():
        aligned[t] = df["close"].reindex(aligned.index)
    for t, df in crypto_frames.items():
        # ffill on the UNION of the equity calendar and the crypto index, then select
        # the equity dates: this carries the freshest weekend crypto close (not the
        # stale Friday close) to Monday, while limit=1 leaves longer gaps as NaN.
        combined = df["close"].reindex(aligned.index.union(df.index)).ffill(limit=1)
        aligned[t] = combined.reindex(aligned.index)

    # ponytail: provenance keys are raw tickers (e.g. "^VIX"); consumers that rename
    # columns (rebalance.py maps "^VIX"->"VIX") must key provenance by the post-rename
    # name. No consumer reads it yet.
    aligned.attrs["provenance"] = provenance
    return aligned


def get_last_trading_day(end: date = date.today()) -> date:
    """Return the most recent US-equity trading day on or before ``end``.

    Skips weekends AND NYSE holidays (holiday set reused from ``live.risk``).
    """
    ts = pd.Timestamp(end)
    while ts.weekday() >= 5 or ts.date() in _market_holidays(ts.year):
        ts -= pd.Timedelta(days=1)
    return ts.date()
