"""Guardrails tests for the live runner (rebalance.py / risk.py / state.py).

Fully offline and deterministic: no Alpaca, no network, no Date.now/random. Covers
the drawdown breaker (real equity vs peak), A+B concentration cap from sleeve
weights, the ticker-level drift gate, the expanded NYSE holiday calendar, and the
peak-equity persistence (write only on non-dry-run success).
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import live.state as state
import scripts.rebalance as rb
from live.risk import RiskGuard


# --- Shared fakes ----------------------------------------------------------
class FakeExecutor:
    """Stand-in for AlpacaExecutor with a controlled book and a rebalance spy."""

    def __init__(self, equity=100_000.0, positions=None, paper=True):
        self.equity = equity
        self._positions = positions or {}
        self.paper = paper
        self.calls = []  # rebalance() call log

    def get_account(self):
        return {
            "equity": self.equity,
            "cash": self.equity * 0.5,
            "buying_power": self.equity,
            "portfolio_value": self.equity,
        }

    def get_positions(self):
        return dict(self._positions)

    def rebalance(self, target, prices, dry_run=True):
        self.calls.append({"target": target, "prices": prices, "dry_run": dry_run})
        return []


def _syn_prices(tickers, end="2025-01-06"):
    idx = pd.to_datetime(["2025-01-02", end])
    return pd.DataFrame({t: [100.0, 101.0] for t in tickers}, index=idx)


def _syn_targets(ticker_weights=None, sleeve_weights=None, tickers=None, end="2025-01-06"):
    tickers = tickers or ["SPY", "TLT", "BIL", "IEF", "GLD"]
    ticker_weights = ticker_weights or {t: 0.2 for t in tickers}
    sleeve_weights = sleeve_weights or pd.Series(
        {"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2}
    )
    return {
        "sleeve_weights": sleeve_weights,
        "ticker_weights": ticker_weights,
        "prices": _syn_prices(tickers, end=end),
        "weights_a_last": pd.Series(dtype=float),
        "weights_b_last": pd.Series(dtype=float),
    }


@pytest.fixture
def patch_paths(tmp_path, monkeypatch):
    """Redirect state.json and the weight log to a temp dir."""
    state_path = tmp_path / "state.json"
    weight_log = tmp_path / "weights.jsonl"
    monkeypatch.setattr(state, "STATE_PATH", state_path)
    monkeypatch.setattr(rb, "WEIGHT_LOG", weight_log)
    return {"state": state_path, "weights": weight_log}


def _seed_peak(path, peak):
    path.write_text(json.dumps({"peak_equity": peak}), encoding="utf-8")


def _read_peak(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("peak_equity")


# --- (a) Drawdown breaker fires on real equity < peak ----------------------
def test_drawdown_breaker_fires_below_threshold():
    # peak 100k, equity 90k -> -10% drawdown; with a 5% breaker it must fire.
    guard = RiskGuard(max_drawdown_pct=-5.0, peak_equity=100_000.0)
    prices = _syn_prices(["SPY", "BIL"])
    res = guard.check(prices, {"SPY": 0.2, "BIL": 0.2}, live_equity=90_000.0,
                      current_date=date(2025, 1, 6))
    assert not res.ok
    assert any("drawdown" in m for m in res.messages)


def test_drawdown_breaker_does_not_fire_near_peak():
    guard = RiskGuard(peak_equity=100_000.0)
    prices = _syn_prices(["SPY", "BIL"])
    res = guard.check(prices, {"SPY": 0.2, "BIL": 0.2}, live_equity=99_000.0,
                      current_date=date(2025, 1, 6))
    assert res.ok
    assert not any("drawdown" in m for m in res.messages)


def test_drawdown_breaker_fires_through_main(monkeypatch, patch_paths, capsys):
    """End-to-end: real equity fetched before guard.check trips the breaker."""
    executor = FakeExecutor(equity=85_000.0)  # 15% below seeded peak -> fires at -10%
    monkeypatch.setattr(rb, "compute_systematic_targets",
                        lambda *a, **k: _syn_targets(end="2025-01-06"))
    monkeypatch.setattr(rb, "AlpacaExecutor", lambda *a, **k: executor)
    _seed_peak(patch_paths["state"], 100_000.0)
    monkeypatch.setattr(rb, "update_peak_equity", lambda e: pytest.fail("peak must not update on a blocked run"))

    monkeypatch.setattr("sys.argv", ["rebalance.py", "--date", "2025-01-06"])
    rc = rb.main()
    assert rc == 2  # RISK BLOCK
    out = capsys.readouterr().out
    assert "drawdown" in out


# --- (b) A+B concentration cap from SLEEVE weights --------------------------
def test_ab_concentration_cap_flags_from_sleeve_weights():
    sleeve = pd.Series({"A": 0.4, "B": 0.4, "rates": 0.1, "bear": 0.1, "cta": 0.0})
    guard = RiskGuard(max_sleeve_pct=0.7)
    prices = _syn_prices(["SPY", "BIL", "TLT", "IEF", "GLD"])
    res = guard.check(prices, {"SPY": 0.1, "BIL": 0.1}, sleeve_weights=sleeve,
                      current_date=date(2025, 1, 6))
    assert not res.ok
    assert any("core sleeve" in m for m in res.messages)


def test_ab_concentration_cap_does_not_misfire_within_cap():
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})
    guard = RiskGuard(max_sleeve_pct=0.7)
    prices = _syn_prices(["SPY", "BIL", "TLT", "IEF", "GLD"])
    res = guard.check(prices, {"SPY": 0.1, "BIL": 0.1}, sleeve_weights=sleeve,
                      current_date=date(2025, 1, 6))
    assert res.ok


def test_ab_concentration_cap_old_ticker_keyed_path_is_zero():
    """Without sleeve_weights the old ticker-keyed lookup yields 0 (never misfires)."""
    guard = RiskGuard(max_sleeve_pct=0.7)
    prices = _syn_prices(["SPY", "BIL"])
    # "A"/"B" keys are sleeve names, not tickers -> 0 + 0 = 0, no false positive.
    res = guard.check(prices, {"SPY": 0.2, "BIL": 0.2, "A": 0.0, "B": 0.0},
                      current_date=date(2025, 1, 6))
    assert res.ok


# --- (c) Ticker-level drift gate -------------------------------------------
def test_drift_gate_skips_when_within_threshold(patch_paths):
    executor = FakeExecutor(equity=100_000.0,
                            positions={t: 20_000.0 for t in ["SPY", "TLT", "BIL", "IEF", "GLD"]})
    targets = {t: 0.2 for t in ["SPY", "TLT", "BIL", "IEF", "GLD"]}
    prices = _syn_prices(list(targets))
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    orders, skipped = rb.decide_and_execute(
        executor, targets, sleeve, prices, date(2025, 3, 3),
        account=executor.get_account(), drift_threshold=0.05, dry_run=True,
    )
    assert skipped is True
    assert orders == []
    assert executor.calls == []  # no rebalance placed

    # Targets are still logged when skipping.
    log = [json.loads(line) for line in patch_paths["weights"].read_text(encoding="utf-8").splitlines()]
    assert log and log[-1]["target_weights"] == targets
    assert log[-1]["orders"] == []


def test_drift_gate_fires_when_beyond_threshold(patch_paths):
    executor = FakeExecutor(equity=100_000.0, positions={"SPY": 20_000.0})  # SPY at 20%
    targets = {"SPY": 0.5, "BIL": 0.5}  # SPY drifts to 50% -> 0.30 > 0.05
    prices = _syn_prices(list(targets))
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    orders, skipped = rb.decide_and_execute(
        executor, targets, sleeve, prices, date(2025, 3, 3),
        account=executor.get_account(), drift_threshold=0.05, dry_run=False,
    )
    assert skipped is False
    assert len(executor.calls) == 1  # rebalance placed


def test_drift_gate_forces_execute_in_annual_window(patch_paths):
    """Even with zero drift, the annual window forces execution."""
    executor = FakeExecutor(equity=100_000.0,
                            positions={t: 20_000.0 for t in ["SPY", "TLT", "BIL", "IEF", "GLD"]})
    targets = {t: 0.2 for t in ["SPY", "TLT", "BIL", "IEF", "GLD"]}
    prices = _syn_prices(list(targets))
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    orders, skipped = rb.decide_and_execute(
        executor, targets, sleeve, prices, date(2025, 1, 5),  # annual window
        account=executor.get_account(), drift_threshold=0.05, dry_run=True,
    )
    assert skipped is False
    assert len(executor.calls) == 1


def test_drift_gate_boundary_at_exact_threshold_skips(patch_paths):
    """max_drift == drift_threshold (default 0.05) -> skipped (<= is inclusive).

    Engineered so the max drift is exactly the float 0.05: BIL current weight is
    5000/100000 == 0.05 (the same float as the literal 0.05) and the BIL target is
    0.0, so |0.0 - 0.05| == 0.05 exactly; SPY is held at target (drift 0.0).
    """
    executor = FakeExecutor(equity=100_000.0, positions={"SPY": 95_000.0, "BIL": 5_000.0})
    targets = {"SPY": 0.95, "BIL": 0.0}
    prices = _syn_prices(list(targets))
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    orders, skipped = rb.decide_and_execute(
        executor, targets, sleeve, prices, date(2025, 3, 3),
        account=executor.get_account(), drift_threshold=0.05, dry_run=True,
    )
    assert skipped is True
    assert orders == []
    assert executor.calls == []


def test_drift_gate_fires_on_current_only_ticker(patch_paths):
    """An extra CURRENT ticker absent from targets counts as drift (union of tickers)."""
    # SPY 20k (20%) is current-only; targets have no SPY -> drift 0.20 > 0.05.
    executor = FakeExecutor(equity=100_000.0, positions={"SPY": 20_000.0})
    targets = {"BIL": 1.0}
    prices = _syn_prices(["BIL", "SPY"])
    sleeve = pd.Series({"A": 0.2, "B": 0.2, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    orders, skipped = rb.decide_and_execute(
        executor, targets, sleeve, prices, date(2025, 3, 3),
        account=executor.get_account(), drift_threshold=0.05, dry_run=True,
    )
    assert skipped is False
    assert len(executor.calls) == 1


# --- (c2) AlpacaExecutor None-client guard (dry-run path) ------------------
def test_executor_get_positions_none_client_returns_empty():
    from live.alpaca_executor import AlpacaExecutor
    # Dry-run construction (client=None) must not crash when the drift gate
    # calls get_positions(); returns an empty book so drift forces a (dry) trade.
    executor = AlpacaExecutor(client=None)
    assert executor.get_positions() == {}


# --- (d) should_run_today holidays ----------------------------------------
@pytest.mark.parametrize("d,expected", [
    (date(2025, 11, 27), False),  # Thanksgiving 2025 (4th Thursday)
    (date(2025, 5, 26), False),   # Memorial Day 2025 (last Monday of May)
    (date(2025, 9, 1), False),    # Labor Day 2025 (1st Monday of Sep)
    (date(2025, 1, 20), False),  # MLK Day 2025 (3rd Monday of Jan)
    (date(2025, 6, 19), False),  # Juneteenth 2025
    (date(2026, 4, 3), False),   # Good Friday 2026
    (date(2025, 1, 4), False),    # Saturday
    (date(2025, 1, 1), False),    # New Year's Day
    (date(2025, 7, 4), False),    # Independence Day
    (date(2025, 12, 25), False),  # Christmas
    (date(2025, 3, 3), True),     # normal Monday
    (date(2025, 6, 18), True),    # normal weekday (day before Juneteenth)
])
def test_should_run_today(d, expected):
    assert RiskGuard().should_run_today(d) is expected


# --- (e) Peak equity persistence ------------------------------------------
def test_peak_not_persisted_on_dry_run(monkeypatch, patch_paths):
    executor = FakeExecutor(equity=100_000.0)
    monkeypatch.setattr(rb, "compute_systematic_targets",
                        lambda *a, **k: _syn_targets(end="2025-03-03"))
    monkeypatch.setattr(rb, "AlpacaExecutor", lambda *a, **k: executor)

    update_calls = []

    def _spy(equity):
        update_calls.append(equity)
        return state.update_peak_equity(equity)  # delegate to real (patched path)

    monkeypatch.setattr(rb, "update_peak_equity", _spy)
    _seed_peak(patch_paths["state"], 100_000.0)

    monkeypatch.setattr("sys.argv", ["rebalance.py", "--date", "2025-03-03", "--dry-run"])
    rc = rb.main()
    assert rc == 0
    assert update_calls == []                      # never called on dry-run
    assert _read_peak(patch_paths["state"]) == 100_000.0  # persisted peak unchanged


def test_peak_persisted_on_non_dry_run_success(monkeypatch, patch_paths):
    executor = FakeExecutor(equity=123_000.0)
    monkeypatch.setattr(rb, "compute_systematic_targets",
                        lambda *a, **k: _syn_targets(end="2025-03-03"))
    monkeypatch.setattr(rb, "AlpacaExecutor", lambda *a, **k: executor)

    update_calls = []

    def _spy(equity):
        update_calls.append(equity)
        return state.update_peak_equity(equity)

    monkeypatch.setattr(rb, "update_peak_equity", _spy)
    _seed_peak(patch_paths["state"], 100_000.0)

    monkeypatch.setattr("sys.argv", ["rebalance.py", "--date", "2025-03-03"])
    rc = rb.main()
    assert rc == 0
    assert update_calls == [123_000.0]                       # called once with real equity
    assert _read_peak(patch_paths["state"]) == 123_000.0     # peak updated to new high
