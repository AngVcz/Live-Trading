"""Offline tests for live/data_feed.py: calendar, ^VIX provenance, per-source
cache + staleness, and holiday-aware get_last_trading_day.

No network: yfinance / Alpaca fetchers are patched with fakes. The cache dir is
redirected to a tmp_path so no real .parquet artifacts are touched.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

import live.data_feed as dfd


# --- shared helpers --------------------------------------------------------
def _ohlcv(idx, close):
    """Build a minimal OHLCV frame (only `close` is used downstream)."""
    idx = pd.DatetimeIndex(pd.to_datetime(idx), name="date")
    n = len(idx)
    return pd.DataFrame(
        {c: [0.0] * n for c in ("open", "high", "low", "volume")},
        index=idx,
    ).assign(close=list(close))


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dfd, "CACHE_DIR", tmp_path)
    return tmp_path


# ==========================================================================
# (a) Panel on equity-calendar intersection; crypto ffill(limit=1)
# ==========================================================================
def test_panel_no_weekend_rows_and_crypto_ffill_limit1(monkeypatch):
    weekdays = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
                "2025-01-13", "2025-01-14"]  # Mon..Fri, then next Mon, Tue
    # SPY: weekdays only (equity calendar).
    spy_close = [100, 101, 102, 103, 104, 105, 106]
    spy_df = _ohlcv(weekdays, spy_close)

    # BTC: trades 7 days/week incl. weekend, but stops after Sunday 01-12
    # (no Mon 01-13 / Tue 01-14 bars -> those should be NaN after ffill limit 1).
    btc_idx = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
               "2025-01-11", "2025-01-12"]  # Fri 01-10, Sat 01-11, Sun 01-12
    btc_close = [200, 201, 202, 203, 204, 205, 206]
    btc_df = _ohlcv(btc_idx, btc_close)

    def fake_fetch_ohlcv(ticker, start, end, prefer_alpaca=True, **kw):
        if ticker == "SPY":
            r = spy_df.copy()
        elif ticker == "BTC-USD":
            r = btc_df.copy()
        else:
            raise ValueError(ticker)
        r.attrs["source"] = "yfinance"
        return r

    monkeypatch.setattr(dfd, "fetch_ohlcv", fake_fetch_ohlcv)
    panel = dfd.fetch_panel(["SPY", "BTC-USD"], date(2025, 1, 6), date(2025, 1, 14),
                            prefer_alpaca=False)

    # No weekend rows.
    assert (panel.index.dayofweek < 5).all()
    # Calendar equals the equity (SPY) weekday index — no Sat/Sun fabricated.
    assert "2025-01-11" not in panel.index
    assert "2025-01-12" not in panel.index

    # Crypto ffilled at most one day: Mon 01-13 carries the freshest weekend close
    # (Sun 01-12 == 206, NOT the stale Fri 01-10 == 204); Tue 01-14 has no recent
    # crypto bar -> NaN (not fabricated).
    assert panel.loc["2025-01-10", "BTC-USD"] == 204
    assert panel.loc["2025-01-13", "BTC-USD"] == 206  # weekend close carried Fri->Mon
    assert pd.isna(panel.loc["2025-01-14", "BTC-USD"])

    # Equity column unaffected.
    assert panel.loc["2025-01-13", "SPY"] == 105


def test_panel_raises_when_only_crypto_frames(monkeypatch):
    btc_df = _ohlcv(["2025-01-06"], [200])

    def fake_fetch_ohlcv(ticker, start, end, prefer_alpaca=True, **kw):
        r = btc_df.copy()
        r.attrs["source"] = "yfinance"
        return r

    monkeypatch.setattr(dfd, "fetch_ohlcv", fake_fetch_ohlcv)
    with pytest.raises(RuntimeError, match="No equity"):
        dfd.fetch_panel(["BTC-USD"], date(2025, 1, 6), date(2025, 1, 10))


def test_panel_partial_equity_failure_warns_and_nan_column(monkeypatch):
    """One of two equity fetchers raising: panel still built, failed ticker
    all-NaN, a warning emitted, no raise."""
    spy_df = _ohlcv(["2025-01-06", "2025-01-07", "2025-01-08"], [100, 101, 102])

    def fake_fetch_ohlcv(ticker, start, end, prefer_alpaca=True, **kw):
        if ticker == "SPY":
            r = spy_df.copy()
        else:
            raise RuntimeError(f"boom for {ticker}")
        r.attrs["source"] = "yfinance"
        return r

    monkeypatch.setattr(dfd, "fetch_ohlcv", fake_fetch_ohlcv)

    with pytest.warns(UserWarning, match="partial fetch failure"):
        panel = dfd.fetch_panel(["SPY", "TLT"], date(2025, 1, 6), date(2025, 1, 8),
                                prefer_alpaca=False)

    assert "SPY" in panel.columns
    assert "TLT" in panel.columns
    assert panel["SPY"].notna().all()
    assert panel["TLT"].isna().all()  # failed ticker -> all-NaN column


# ==========================================================================
# (b) ^VIX resolves via yfinance; Alpaca not called; provenance recorded
# ==========================================================================
def test_vix_uses_yfinance_not_alpaca(monkeypatch):
    vix_idx = ["2025-01-06", "2025-01-07", "2025-01-08"]
    vix_df = _ohlcv(vix_idx, [18.0, 18.5, 19.0])

    alpaca_spy = MagicMock(side_effect=AssertionError("Alpaca must not be called for ^VIX"))
    monkeypatch.setattr(dfd, "_fetch_alpaca", alpaca_spy)

    def fake_yf(ticker, start, end):
        assert ticker == "^VIX"
        return vix_df.copy()

    monkeypatch.setattr(dfd, "_fetch_yfinance", fake_yf)

    out = dfd.fetch_ohlcv("^VIX", date(2025, 1, 6), date(2025, 1, 8), prefer_alpaca=True)
    assert out.attrs["source"] == "yfinance"
    alpaca_spy.assert_not_called()

    # Through fetch_panel, provenance dict records ^VIX -> yfinance.
    monkeypatch.setattr(dfd, "fetch_ohlcv", lambda *a, **k: out)
    panel = dfd.fetch_panel(["^VIX"], date(2025, 1, 6), date(2025, 1, 8), prefer_alpaca=True)
    assert panel.attrs["provenance"]["^VIX"] == "yfinance"


def test_alpaca_skip_for_crypto_and_indices(monkeypatch):
    """BTC-USD and ^VIX both skip Alpaca; a tradeable equity does not."""
    assert dfd._skip_alpaca("^VIX")
    assert dfd._skip_alpaca("BTC-USD")
    assert dfd._skip_alpaca("ETH-USD")
    assert not dfd._skip_alpaca("SPY")
    assert not dfd._skip_alpaca("TLT")


# ==========================================================================
# (c) Per-source cache key + staleness tail refresh
# ==========================================================================
def _spy_df(start="2025-01-06", end="2025-01-10", base=100.0):
    idx = pd.bdate_range(start, end)  # weekdays
    return _ohlcv(idx, [base + i for i in range(len(idx))])


def test_per_source_cache_files_are_distinct(monkeypatch):
    """Fetching the same ticker from two sources produces two distinct cache files."""
    # Historical end -> not live-relevant -> no tail refresh -> one fetch each.
    monkeypatch.setattr(dfd, "_today", lambda: date(2025, 1, 15))
    start, end = date(2024, 1, 1), date(2024, 1, 10)

    alpaca_df = _spy_df(start="2024-01-01", end="2024-01-10", base=100.0)
    yf_df = _spy_df(start="2024-01-01", end="2024-01-10", base=200.0)
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda t, s, e, **k: alpaca_df.copy())
    monkeypatch.setattr(dfd, "_fetch_yfinance",
                        lambda t, s, e: yf_df.copy())

    dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=True)
    dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)

    assert (dfd.CACHE_DIR / "SPY__alpaca.parquet").exists()
    assert (dfd.CACHE_DIR / "SPY__yfinance.parquet").exists()


def test_mismatched_source_cache_is_not_reused(monkeypatch):
    """A cache written under source A is not served when source B is preferred."""
    monkeypatch.setattr(dfd, "_today", lambda: date(2025, 1, 15))
    start, end = date(2024, 1, 1), date(2024, 1, 10)

    # First call: yfinance only (alpaca raises). Writes SPY__yfinance.parquet.
    yf_df = _spy_df(start="2024-01-01", end="2024-01-10", base=200.0)
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no alpaca")))
    yf_calls = []
    monkeypatch.setattr(dfd, "_fetch_yfinance",
                        lambda t, s, e: yf_calls.append(1) or yf_df.copy())

    dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)
    assert (dfd.CACHE_DIR / "SPY__yfinance.parquet").exists()
    assert not (dfd.CACHE_DIR / "SPY__alpaca.parquet").exists()

    # Second call: prefer_alpaca=True but Alpaca returns real data this time.
    # The yfinance cache must NOT be served (different source key).
    alpaca_df = _spy_df(start="2024-01-01", end="2024-01-10", base=300.0)
    alpaca_calls = []
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda t, s, e, **k: alpaca_calls.append(1) or alpaca_df.copy())

    out = dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=True)
    assert alpaca_calls, "Alpaca must be fetched (yfinance cache must not be reused)"
    assert out.attrs["source"] == "alpaca"
    assert (dfd.CACHE_DIR / "SPY__alpaca.parquet").exists()


def test_stale_cache_hit_triggers_tail_refresh(monkeypatch):
    """A live-relevant cache hit whose tail is stale triggers a refresh fetch."""
    now = date(2025, 1, 8)  # Wednesday
    monkeypatch.setattr(dfd, "_today", lambda: now)
    start, end = date(2025, 1, 6), date(2025, 1, 6)  # Monday requested

    # Seed a cache (source=yfinance) covering up to Monday 01-06 (== end -> hit),
    # but its last bar (Mon) is 2 days behind `now` (Wed) -> stale.
    seed = _ohlcv(["2025-01-06"], [100.0])
    dfd._save_cache("SPY", "yfinance", seed)

    refreshed = []
    def fake_yf(ticker, s, e):
        refreshed.append((s, e))
        return _ohlcv(["2025-01-07", "2025-01-08"], [101.0, 102.0])
    monkeypatch.setattr(dfd, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))

    out = dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)
    assert refreshed, "stale hit must trigger a tail refresh fetch"
    # Refresh fetches from the day after cache's last bar up to today.
    assert refreshed[0][0] == date(2025, 1, 7)
    assert refreshed[0][1] == now
    assert out.attrs["source"] == "yfinance"
    # Cache extended with the refreshed tail.
    cached = dfd._load_cache("SPY", "yfinance")
    assert cached.index.max() == pd.Timestamp("2025-01-08")


def test_stale_cache_hit_served_when_refresh_fails(monkeypatch):
    """A range-covering cache hit must be served even if the tail-refresh fetch
    raises — no RuntimeError, no silent source switch."""
    now = date(2025, 1, 8)  # Wednesday
    monkeypatch.setattr(dfd, "_today", lambda: now)
    start, end = date(2025, 1, 6), date(2025, 1, 6)  # Monday requested

    # Range-covering cache (last bar == end == Mon 01-06, 2 days behind now -> stale).
    seed = _ohlcv(["2025-01-06"], [100.0])
    dfd._save_cache("SPY", "yfinance", seed)

    # Tail-refresh fetcher raises (network blip). Must NOT switch to Alpaca.
    def failing_yf(*a, **k):
        raise RuntimeError("network blip on refresh")
    monkeypatch.setattr(dfd, "_fetch_yfinance", failing_yf)
    alpaca_spy = MagicMock(side_effect=AssertionError(
        "stale-but-covering cache must not switch to fallback source"))
    monkeypatch.setattr(dfd, "_fetch_alpaca", alpaca_spy)

    out = dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)
    assert out.attrs["source"] == "yfinance"  # no source switch
    assert out.loc["2025-01-06", "close"] == 100.0
    alpaca_spy.assert_not_called()


def test_fresh_cache_hit_does_not_refresh(monkeypatch):
    now = date(2025, 1, 8)  # Wednesday
    monkeypatch.setattr(dfd, "_today", lambda: now)
    start, end = date(2025, 1, 6), date(2025, 1, 6)  # Monday requested

    # Seed cache covering up to Wed 01-08 (== now -> fresh, within staleness 1).
    seed = _ohlcv(["2025-01-06", "2025-01-07", "2025-01-08"], [100.0, 101.0, 102.0])
    dfd._save_cache("SPY", "yfinance", seed)

    def fake_yf(*a, **k):
        raise AssertionError("fresh hit must not fetch")
    monkeypatch.setattr(dfd, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))

    out = dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)
    assert out.attrs["source"] == "yfinance"
    assert out.loc["2025-01-06", "close"] == 100.0


def test_backtest_cache_hit_not_refreshed(monkeypatch):
    """A historical (non-live-relevant) cache hit must not trigger a tail refresh."""
    now = date(2025, 7, 31)
    monkeypatch.setattr(dfd, "_today", lambda: now)
    start, end = date(2024, 1, 1), date(2024, 1, 10)  # far from now

    seed = _spy_df(start="2024-01-01", end="2024-01-10", base=100.0)
    dfd._save_cache("SPY", "yfinance", seed)

    def fake_yf(*a, **k):
        raise AssertionError("backtest hit must not fetch")
    monkeypatch.setattr(dfd, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(dfd, "_fetch_alpaca",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))

    out = dfd.fetch_ohlcv("SPY", start, end, prefer_alpaca=False)
    assert out.attrs["source"] == "yfinance"


# ==========================================================================
# (d) get_last_trading_day skips NYSE holidays
# ==========================================================================
@pytest.mark.parametrize("d,expected", [
    (date(2026, 4, 3), date(2026, 4, 2)),   # Good Friday 2026 -> Thu Apr 2
    (date(2025, 11, 27), date(2025, 11, 26)),  # Thanksgiving 2025 -> Wed Nov 26
    (date(2025, 11, 28), date(2025, 11, 28)),  # Fri after Thanksgiving (normal) -> itself
    (date(2025, 5, 26), date(2025, 5, 23)),  # Memorial Day 2025 (Mon) -> Fri May 23
    (date(2025, 1, 4), date(2025, 1, 3)),    # Saturday -> Fri Jan 3
    (date(2025, 1, 5), date(2025, 1, 3)),    # Sunday -> Fri Jan 3
    (date(2025, 1, 6), date(2025, 1, 6)),    # normal Monday -> itself
    (date(2025, 3, 3), date(2025, 3, 3)),    # normal Monday -> itself
    (date(2025, 1, 1), date(2024, 12, 31)),  # New Year's Day 2025 (Wed) -> Tue Dec 31 2024
    (date(2025, 7, 4), date(2025, 7, 3)),     # Independence Day 2025 (Fri holiday) -> Thu Jul 3
    (date(2025, 12, 25), date(2025, 12, 24)),  # Christmas 2025 (Thu) -> Wed Dec 24
])
def test_get_last_trading_day(d, expected):
    assert dfd.get_last_trading_day(d) == expected