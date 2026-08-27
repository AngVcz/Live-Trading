"""Smoke tests for the live A+B+Diversifier Sleeves pipeline.

These tests do NOT call Alpaca; they use cached yfinance data and verify that the
signal engine, sleeve builder, portfolio construction, and risk guard produce sane
outputs end-to-end.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest


def _load_prices() -> pd.DataFrame:
    """Build a small cached price panel for smoke testing."""
    from live.data_feed import fetch_panel
    from live.core_signals import UNIVERSE as CORE_UNIVERSE
    end = date(2025, 6, 30)
    start = end - timedelta(days=365 * 4)
    tickers = list(CORE_UNIVERSE) + [
        "TLT", "IEF", "GLD", "PDBC", "KMLM", "DBMF", "VIXM", "SH", "BIL", "VIXY", "PSQ",
        "^VIX",
    ]
    return fetch_panel(tickers, start, end, prefer_alpaca=False)


def test_core_signals_shape():
    from live.core_signals import build_core_returns
    prices = _load_prices()
    prices = prices.rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    assert len(ret_a) == len(ret_b) > 252
    assert not ret_a.isna().all()
    assert not ret_b.isna().all()
    assert weights_a.shape[1] == weights_b.shape[1]
    sum_a = float(abs(weights_a.iloc[-1]).sum())
    valid = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    assert any(abs(sum_a - v) < 1e-6 for v in valid)


def test_sleeve_returns():
    from live.portfolio import build_sleeve_returns
    prices = _load_prices()
    sleeves = build_sleeve_returns(prices)
    assert list(sleeves.columns) == ["rates", "bear", "cta"]
    assert not sleeves.isna().all().any()


def test_portfolio_weights_sum_to_one():
    from live.core_signals import build_core_returns
    from live.portfolio import (
        SleeveConfig,
        build_live_weights,
        build_sleeve_returns,
        decompose_target_to_tickers,
    )
    prices = _load_prices()
    prices = prices.rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve_rets = build_sleeve_returns(prices)
    config = SleeveConfig()
    sleeve_weights = build_live_weights(ret_a, ret_b, sleeve_rets, config)
    assert abs(sleeve_weights.sum() - 1.0) < 1e-6
    # Default config now disables bear; confirm the freed 20% is routed to BIL ballast.
    assert abs(sleeve_weights.get("BIL_ballast", 0.0) - 0.2) < 1e-9
    assert abs(sleeve_weights.get("bear", 0.0)) < 1e-9

    target_tickers = decompose_target_to_tickers(
        sleeve_weights, weights_a.iloc[-1], weights_b.iloc[-1], prices
    )
    assert abs(sum(target_tickers.values()) - 1.0) < 1e-6
    assert all(w >= -1e-6 for w in target_tickers.values())
    # BIL carries at least the freed bear budget (more if A/B residuals also floor to BIL).
    assert target_tickers.get("BIL", 0.0) >= 0.2 - 1e-6


def test_risk_guard_blocks_stale_data():
    from live.risk import RiskGuard
    guard = RiskGuard(stale_data_days=2)
    prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    result = guard.check(prices, {"SPY": 0.5}, current_date=date(2020, 1, 10))
    assert not result.ok
    assert any("stale" in m for m in result.messages)


def test_risk_guard_blocks_weekend():
    from live.risk import RiskGuard
    guard = RiskGuard()
    assert not guard.should_run_today(date(2025, 1, 4))  # Saturday
    assert guard.should_run_today(date(2025, 1, 6))      # Monday


def test_executor_dry_run():
    from live.alpaca_executor import AlpacaExecutor, TargetPortfolio
    executor = AlpacaExecutor(client=None, fractional=True)
    target = TargetPortfolio(
        date=pd.Timestamp("2025-01-06"),
        targets={"SPY": 5000.0, "TLT": 5000.0},
        expected_cash=0.0,
    )
    prices = {"SPY": 500.0, "TLT": 100.0}
    results = executor.rebalance(target, prices, dry_run=True)
    spies = [r for r in results if r.ticker == "SPY"]
    assert spies and spies[0].qty == pytest.approx(10.0, 0.01)


def test_tilt_options_bounds():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns
    from live.discretionary import build_tilt_options, OPTION_NAMES
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    options = build_tilt_options(sleeve, weights_a.iloc[-1], weights_b.iloc[-1], prices)
    assert set(options) == set(OPTION_NAMES)
    ab = {}
    for name in OPTION_NAMES:
        o = options[name]
        assert abs(sum(o["tickers"].values()) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in o["tickers"].values())
        assert all(w <= 0.35 + 1e-9 for t, w in o["tickers"].items() if t != "BIL")
        for k in ("A", "B", "rates", "cta"):
            assert o["sleeve"][k] <= 0.45 + 1e-9, (name, k)
        ab[name] = o["sleeve"]["A"] + o["sleeve"]["B"]
    assert ab["risk_off"] < ab["systematic"] < ab["risk_on"]
    assert ab["risk_on"] <= 0.70 + 1e-9


def test_tilt_options_vix_overlay_disables_risk_on():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns
    from live.discretionary import build_tilt_options
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    options = build_tilt_options(sleeve, weights_a.iloc[-1], weights_b.iloc[-1],
                                 prices, vix_overlay_active=True)
    diff = sum(abs(options["risk_on"]["sleeve"][k] - options["systematic"]["sleeve"][k])
               for k in options["systematic"]["sleeve"])
    assert diff < 1e-12
    assert "disabled" in options["risk_on"]["note"]


def test_tilt_edges_unfundable_and_caps():
    from live.discretionary import _tilt_risk_on, _tilt_risk_off, _enforce_caps, _clip_tickers
    # (a) unfundable risk-on -> identity + note
    s = pd.Series({"A": 0.3, "B": 0.3, "rates": 0.05, "BIL_ballast": 0.05, "cta": 0.30})
    out, note = _tilt_risk_on(s)
    assert out.equals(s) and "unfundable" in note
    # (b) cap repair preserves the sum and notes both repairs
    s2 = pd.Series({"A": 0.60, "B": 0.20, "rates": 0.10, "BIL_ballast": 0.05, "cta": 0.05})
    out2, note2 = _enforce_caps(s2)
    assert abs(out2.sum() - 1.0) < 1e-9
    assert out2["A"] + out2["B"] <= 0.70 + 1e-9
    assert all(out2[k] <= 0.45 + 1e-9 for k in ("A", "B", "rates", "cta"))
    assert "70%" in note2 and "45%" in note2
    # (c) risk-off: rates cap binding vs no-uptrend routing
    base = pd.Series({"A": 0.20, "B": 0.20, "rates": 0.20, "BIL_ballast": 0.20, "cta": 0.20})
    up, _ = _tilt_risk_off(base, rates_in_uptrend=True)
    down, _ = _tilt_risk_off(base, rates_in_uptrend=False)
    assert abs(up["rates"] - 0.45) < 1e-9 and abs(up["BIL_ballast"] - 0.25) < 1e-9
    assert abs(down["rates"] - 0.20) < 1e-9 and abs(down["BIL_ballast"] - 0.50) < 1e-9
    assert abs(up.sum() - 1.0) < 1e-9 and abs(down.sum() - 1.0) < 1e-9
    # (d) ticker clip spills to BIL
    clipped = _clip_tickers({"SPY": 0.60, "BIL": 0.40})
    assert abs(clipped["SPY"] - 0.35) < 1e-12 and abs(clipped["BIL"] - 0.65) < 1e-12


def test_metrics_panel_smoke():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns, decompose_target_to_tickers
    from live.morning_metrics import compute_metrics_panel
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    tickers = decompose_target_to_tickers(sleeve, weights_a.iloc[-1], weights_b.iloc[-1], prices)
    m = compute_metrics_panel(prices, tickers, weights_a.iloc[-1], weights_b.iloc[-1],
                              equity=100_000.0, as_of=date(2025, 6, 30))
    for key in ("as_of", "vix_close", "vix_change_1d", "vix_percentile_252d",
                "vix_overlay_active", "tnx_10y_level", "tnx_change_5d",
                "tlt_above_sma200", "ief_above_sma200", "breadth_pct_above_sma200",
                "holdings_below_sma200", "equity", "peak_equity", "drawdown_pct",
                "guardrail_margin_pct", "turnover_oneway_pct", "sleeve_weights",
                "top_tickers", "macro_calendar"):
        assert key in m, key
    assert isinstance(m["vix_overlay_active"], bool)
    if m["vix_percentile_252d"] != "n/a":
        assert 0.0 <= m["vix_percentile_252d"] <= 1.0
    assert isinstance(m["holdings_below_sma200"], list)
    assert m["macro_calendar"] == []
