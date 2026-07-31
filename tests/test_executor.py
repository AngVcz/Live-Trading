"""Execution-hardening tests for live/alpaca_executor.py.

Deterministic and offline: a fake Alpaca client records submitted
MarketOrderRequest/LimitOrderRequest objects; no network, no real orders.
Covers: DAY time-in-force, $1 min-notional skip, same-day log APPEND,
illiquid limit-price with market fallback, wash-sale WARN-only tracking,
and the buying-power re-fetch TODO being honored.
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from live.alpaca_executor import (
    AlpacaExecutor,
    TargetPortfolio,
    WASH_SALE_PATH,
    record_realized_loss,
    _wash_sale_loss_within,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

import live.alpaca_executor as exe_mod


# --- Shared fakes ----------------------------------------------------------
class FakeAccount:
    def __init__(self, equity=100_000.0, cash=50_000.0, buying_power=100_000.0):
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power
        self.portfolio_value = equity


class FakePosition:
    def __init__(self, symbol, market_value):
        self.symbol = symbol
        self.market_value = market_value


class FakeSubmitted:
    def __init__(self, status="filled"):
        self.status = status


class FakeClient:
    """Records submitted order requests; no network."""

    def __init__(self, positions=None, account=None, submit_side_effect=None):
        self._paper = True
        self._positions = positions or []
        self._account = account or FakeAccount()
        self.submitted = []  # list of submitted request objects
        self.submit_side_effect = submit_side_effect  # callable(req) -> may raise
        self.get_account_calls = 0

    def get_account(self):
        self.get_account_calls += 1
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def submit_order(self, req):
        self.submitted.append(req)
        if self.submit_side_effect is not None:
            return self.submit_side_effect(req)
        return FakeSubmitted()


# --- Fixtures --------------------------------------------------------------
@pytest.fixture
def wash_sale_path(tmp_path, monkeypatch):
    p = tmp_path / "wash_sale.json"
    monkeypatch.setattr(exe_mod, "WASH_SALE_PATH", p)
    return p


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    d = tmp_path / "orders"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(exe_mod, "LOG_DIR", d)
    return d


def _target(tickers_dollars, as_of="2025-01-06", cash=0.0):
    return TargetPortfolio(
        date=pd.Timestamp(as_of),
        targets=dict(tickers_dollars),
        expected_cash=cash,
    )


# --- (a) DAY time-in-force for whole-share and fractional -------------------
def test_whole_share_market_order_uses_DAY_tif():
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=False)
    # SPY 5000 @ 100 -> 50 whole shares.
    executor.rebalance(
        _target({"SPY": 5000.0}, cash=95_000.0),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert len(client.submitted) == 1
    req = client.submitted[0]
    assert isinstance(req, MarketOrderRequest)
    assert req.time_in_force == TimeInForce.DAY
    assert req.side == OrderSide.BUY
    assert req.qty == Decimal("50")


def test_fractional_market_order_keeps_DAY_tif():
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    # SPY 1500 @ 100 -> 15.0 fractional qty (whole-share would be 15 too), so
    # use a non-integer qty: 1234 @ 100 -> 12.34 shares.
    executor.rebalance(
        _target({"SPY": 1234.0}, cash=98_766.0),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert len(client.submitted) == 1
    req = client.submitted[0]
    assert isinstance(req, MarketOrderRequest)
    assert req.time_in_force == TimeInForce.DAY
    assert float(req.qty) == pytest.approx(12.34, abs=1e-6)


# --- (b) $1 min-notional skip vs $0.01 delta deadband -----------------------
def test_sub_one_dollar_notional_is_skipped():
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    # delta 0.50: clears the $0.01 deadband but below the $1 min-notional gate.
    executor.rebalance(
        _target({"SPY": 0.50}, cash=99_999.50),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert client.submitted == []  # no order sent


def test_just_above_one_dollar_with_real_delta_is_submitted():
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    # delta 1.50: clears both the $0.01 deadband and the $1 min-notional gate.
    executor.rebalance(
        _target({"SPY": 1.50}, cash=99_998.50),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert len(client.submitted) == 1
    assert client.submitted[0].side == OrderSide.BUY


def test_delta_deadband_below_one_cent_skips():
    """A delta < $0.01 is skipped by the delta deadband (not the $1 gate)."""
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    executor.rebalance(
        _target({"SPY": 0.005}, cash=99_999.995),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert client.submitted == []


# --- (c) same-day log APPEND (not overwrite) -------------------------------
def test_same_day_rebalance_appends_log(log_dir):
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    # Run 1: BUY SPY.
    executor.rebalance(
        _target({"SPY": 100.0}, cash=99_900.0, as_of="2025-03-03"),
        {"SPY": 100.0},
        dry_run=False,
    )
    # Run 2: BUY TLT on the same day.
    executor.rebalance(
        _target({"TLT": 100.0}, cash=99_900.0, as_of="2025-03-03"),
        {"TLT": 100.0},
        dry_run=False,
    )
    path = log_dir / "orders_20250303.csv"
    assert path.exists()
    df = pd.read_csv(path)
    # Both runs present (append, not overwrite): 1 row from each run.
    assert set(df["ticker"]) == {"SPY", "TLT"}
    assert len(df) == 2


def test_different_days_get_separate_files(log_dir):
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    executor.rebalance(_target({"SPY": 100.0}, cash=99_900.0, as_of="2025-03-03"),
                       {"SPY": 100.0}, dry_run=False)
    executor.rebalance(_target({"TLT": 100.0}, cash=99_900.0, as_of="2025-03-04"),
                       {"TLT": 100.0}, dry_run=False)
    assert (log_dir / "orders_20250303.csv").exists()
    assert (log_dir / "orders_20250304.csv").exists()


# --- (d) limit price for illiquid set, market fallback ---------------------
def test_illiquid_symbol_gets_limit_order_with_buffer():
    client = FakeClient()
    executor = AlpacaExecutor(
        client=client, fractional=True,
        illiquid_symbols={"KMLM"}, illiquid_buffer=0.005,
    )
    executor.rebalance(
        _target({"KMLM": 1000.0}, cash=99_000.0),
        {"KMLM": 100.0},
        dry_run=False,
    )
    assert len(client.submitted) == 1
    req = client.submitted[0]
    assert isinstance(req, LimitOrderRequest)
    assert req.time_in_force == TimeInForce.DAY
    assert req.side == OrderSide.BUY
    # BUY limit = close * (1 - 0.005) = 99.5
    assert req.limit_price == pytest.approx(99.5, abs=1e-6)


def test_liquid_symbol_stays_plain_market_order():
    client = FakeClient()
    executor = AlpacaExecutor(
        client=client, fractional=True,
        illiquid_symbols={"KMLM"}, illiquid_buffer=0.005,
    )
    executor.rebalance(
        _target({"SPY": 1000.0}, cash=99_000.0),
        {"SPY": 100.0},
        dry_run=False,
    )
    assert len(client.submitted) == 1
    assert isinstance(client.submitted[0], MarketOrderRequest)


def test_illiquid_limit_rejected_falls_back_to_market():
    def side_effect(req):
        if isinstance(req, LimitOrderRequest):
            raise RuntimeError("fractional+limit+DAY not supported for this symbol")
        return FakeSubmitted()

    client = FakeClient(submit_side_effect=side_effect)
    executor = AlpacaExecutor(
        client=client, fractional=True,
        illiquid_symbols={"KMLM"}, illiquid_buffer=0.005,
    )
    executor.rebalance(
        _target({"KMLM": 1000.0}, cash=99_000.0),
        {"KMLM": 100.0},
        dry_run=False,
    )
    # First attempt was a limit order; fallback submitted a market order.
    assert len(client.submitted) == 2
    assert isinstance(client.submitted[0], LimitOrderRequest)
    assert isinstance(client.submitted[1], MarketOrderRequest)


# --- (e) wash-sale 30-day tracker (WARN only) ------------------------------
def test_wash_sale_warns_on_buy_within_30_days(wash_sale_path):
    record_realized_loss("SH", date(2025, 1, 10))
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    with pytest.warns(UserWarning, match="wash-sale: BUY SH within 30d"):
        executor.rebalance(
            _target({"SH": 1000.0}, cash=99_000.0, as_of="2025-01-30"),  # 20d later
            {"SH": 100.0},
            dry_run=False,
        )
    # Order still submitted (warn, not block).
    assert len(client.submitted) == 1


def test_wash_sale_does_not_warn_after_30_days(wash_sale_path):
    record_realized_loss("SH", date(2025, 1, 10))
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        executor.rebalance(
            _target({"SH": 1000.0}, cash=99_000.0, as_of="2025-02-10"),  # 31d later
            {"SH": 100.0},
            dry_run=False,
        )
    assert not any("wash-sale" in str(w.message) for w in caught)
    assert len(client.submitted) == 1


def test_wash_sale_does_not_warn_for_symbol_with_no_prior_loss(wash_sale_path):
    client = FakeClient()
    executor = AlpacaExecutor(client=client, fractional=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        executor.rebalance(
            _target({"SPY": 1000.0}, cash=99_000.0, as_of="2025-01-30"),
            {"SPY": 100.0},
            dry_run=False,
        )
    assert not any("wash-sale" in str(w.message) for w in caught)
    assert len(client.submitted) == 1


def test_wash_sale_sell_records_loss_then_buy_warns(wash_sale_path):
    """Integration: a SELL of SH records a realized-loss date; a later BUY warns."""
    client = FakeClient(positions=[FakePosition("SH", 5_000.0)])
    executor = AlpacaExecutor(client=client, fractional=True)
    # SELL SH on 2025-01-10 (current 5000 -> target 0).
    executor.rebalance(
        _target({"SH": 0.0}, cash=105_000.0, as_of="2025-01-10"),
        {"SH": 100.0},
        dry_run=False,
    )
    state = json.loads(wash_sale_path.read_text(encoding="utf-8"))
    assert "2025-01-10" in state.get("SH", [])

    # BUY SH 15 days later -> warns. Fresh client with no SH position so this
    # is a real BUY (not another SELL).
    buy_client = FakeClient()
    buy_executor = AlpacaExecutor(client=buy_client, fractional=True)
    with pytest.warns(UserWarning, match="wash-sale: BUY SH within 30d"):
        buy_executor.rebalance(
            _target({"SH": 1000.0}, cash=99_000.0, as_of="2025-01-25"),
            {"SH": 100.0},
            dry_run=False,
        )


def test_wash_sale_helper_symbol_not_tracked_returns_none(wash_sale_path):
    record_realized_loss("SPY", date(2025, 1, 10))  # no-op (not tracked)
    assert _wash_sale_loss_within("SPY", date(2025, 1, 20)) is None
    assert not wash_sale_path.exists()


# --- (f) buying-power re-fetch is NOT implemented (TODO honored) ----------
def test_no_second_buying_power_fetch_between_sell_and_buy():
    """Two SELLs then two BUYs: get_account must be called exactly once
    (the pre-rebalance fetch). No half-baked re-fetch between batches."""
    client = FakeClient(
        positions=[
            FakePosition("SPY", 40_000.0),   # to be sold
            FakePosition("TLT", 40_000.0),    # to be sold
        ],
    )
    executor = AlpacaExecutor(client=client, fractional=True)
    executor.rebalance(
        _target({"BIL": 50_000.0, "GLD": 30_000.0}, cash=0.0, as_of="2025-03-03"),
        {"SPY": 100.0, "TLT": 100.0, "BIL": 100.0, "GLD": 100.0},
        dry_run=False,
    )
    assert client.get_account_calls == 1  # only the single pre-rebalance fetch
    # SELLs first, then BUYs.
    submitted = client.submitted
    assert len(submitted) == 4
    assert all(req.side == OrderSide.SELL for req in submitted[:2])
    assert all(req.side == OrderSide.BUY for req in submitted[2:])