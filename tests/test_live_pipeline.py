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
