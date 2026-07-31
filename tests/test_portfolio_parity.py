"""Parity/realism tests for live/portfolio.py.

Synthetic, deterministic, no network. Proves the backtest sleeve path and the
live ``decompose_target_to_tickers`` path agree and that the realism fixes
(cash ballast, gate shift(1), N=2 hysteresis, CTA parity, sleeve costs) hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Fixed date index helper -----------------------------------------------------
def _dates(n: int, start: str = "2019-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _flat_panel(n: int, tickers, base: float = 100.0) -> pd.DataFrame:
    """Constant price panel (all returns 0) for a clean baseline."""
    return pd.DataFrame(
        {t: np.full(n, base) for t in tickers}, index=_dates(n)
    )


# Required tickers for build_sleeve_returns to not raise.
SLEEVE_TICKERS = ["SPY", "TLT", "IEF", "BIL", "SH", "PDBC", "DBMF", "KMLM"]


# (a) residual -> BIL: A's raw weights sum to <1, residual lands in BIL, sum=1.
def test_cash_ballast_residual_to_bil():
    from live.portfolio import decompose_target_to_tickers

    prices = _flat_panel(260, SLEEVE_TICKERS)
    # VIX 0.5 cut: A holds 2 tickers at 0.25 each -> raw_sum 0.5, residual 0.5.
    weight_a = pd.Series({"weight_SPY": 0.25, "weight_QQQ": 0.25})
    weight_b = pd.Series({"weight_TLT": 1.0})
    target = pd.Series({"A": 0.3, "B": 0.1, "rates": 0.2, "bear": 0.2, "cta": 0.2})

    out = decompose_target_to_tickers(target, weight_a, weight_b, prices)

    # A budget 0.3: 0.15 in SPY+QQQ (via weights), 0.15 residual to BIL.
    assert abs(out["SPY"] - 0.3 * 0.25) < 1e-9
    assert abs(out["QQQ"] - 0.3 * 0.25) < 1e-9
    bil_from_a = 0.3 * 0.5
    # B budget 0.1: fully deployed in TLT (raw_sum 1.0), no residual.
    assert abs(out["TLT"] - 0.1) < 1e-9
    # BIL carries A's residual + the rates/bear/cta cash floors (all flat -> BIL).
    assert out["BIL"] >= bil_from_a - 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert all(w >= -1e-9 for w in out.values())


# (b) empty A -> whole A sleeve to BIL.
def test_empty_sleeve_to_bil():
    from live.portfolio import decompose_target_to_tickers

    prices = _flat_panel(260, SLEEVE_TICKERS)
    weight_a = pd.Series(dtype=float)  # empty
    weight_b = pd.Series(dtype=float)  # empty
    target = pd.Series({"A": 0.5, "B": 0.1, "rates": 0.1, "bear": 0.1, "cta": 0.2})

    out = decompose_target_to_tickers(target, weight_a, weight_b, prices)

    # With all prices flat, rates/bear/cta also floor to BIL, so the entire
    # portfolio sits in BIL. The sum==1.0 check is the load-bearing assertion:
    # if empty A/B were erased (the old bug) the sum would be 0.5, not 1.0.
    assert abs(out["BIL"] - 1.0) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


# (c) gate shift: a regime flip on day T affects the sleeve return on T+1, not T.
def test_gate_shift_delays_regime_flip_one_day():
    from live.portfolio import _bear_sleeve, _bear_weights

    n = 260
    prices = _flat_panel(n, SLEEVE_TICKERS).copy()
    # SPY: flat at 100 through day 239, then 95 from day 240 onward.
    prices.loc[:, "SPY"] = 100.0
    prices.iloc[240:, prices.columns.get_loc("SPY")] = 95.0
    # SH: flat at 100, then +10% on the day AFTER the flip (day 242) so the only
    # way the sleeve return picks it up is via the shifted gate.
    prices.loc[:, "SH"] = 100.0
    prices.iloc[242, prices.columns.get_loc("SH")] = 110.0
    prices.loc[:, "BIL"] = 100.0  # zero return

    rets = prices.pct_change(fill_method=None)
    w = _bear_weights(prices)

    # Regime flips to bear at day 241 (2 consecutive closes below SMA).
    assert w.iloc[240]["SH"] == 0.0  # still bull on the day of the 1st below-close
    assert w.iloc[241]["SH"] == 1.0  # flip after the 2nd consecutive below-close

    sleeve = _bear_sleeve(prices, rets)
    # Day 241 uses w[240] (bull -> BIL, 0 return); the flip must NOT show on day 241.
    assert abs(sleeve.iloc[241] - 0.0) < 1e-9
    # Day 242 uses w[241] (bear -> SH) and earns the SH return -> proves shift(1).
    assert abs(sleeve.iloc[242] - 0.10) < 1e-9


# (d) hysteresis: ONE whipsaw close across the SMA does NOT flip; TWO consecutive DO.
def test_hysteresis_whipsaw_needs_two_consecutive():
    from live.portfolio import _hysteresis_regime, HYSTERESIS_N

    assert HYSTERESIS_N == 2
    # condition: F, F, T, F, T, T, F, F
    cond = pd.Series([False, False, True, False, True, True, False, False])
    regime = _hysteresis_regime(cond)
    # Single T at idx2 -> no flip. Single T at idx4 -> no flip. TT at idx4-5 -> flip
    # at idx5. Single F at idx6 -> no flip. FF at idx6-7 -> flip back at idx7.
    expected = [False, False, False, False, False, True, True, False]
    assert regime.tolist() == expected


# (d-sleeve) hysteresis at the sleeve level: one whipsaw close across SMA200
# does not move the sleeve; two consecutive closes do.
def test_hysteresis_sleeve_level():
    from live.portfolio import _bear_weights

    n = 260
    prices = _flat_panel(n, ["SPY", "SH", "BIL"]).copy()
    prices.loc[:, "SPY"] = 100.0
    # One whipsaw: below at day 250 only, back to 100 afterward.
    prices.iloc[250, prices.columns.get_loc("SPY")] = 95.0
    w = _bear_weights(prices)
    assert w.iloc[250]["SH"] == 0.0  # one below-close -> still bull
    assert w.iloc[251]["SH"] == 0.0

    # Two consecutive below: days 252, 253 -> flip at 253.
    prices.iloc[252, prices.columns.get_loc("SPY")] = 95.0
    prices.iloc[253, prices.columns.get_loc("SPY")] = 95.0
    w = _bear_weights(prices)
    assert w.iloc[252]["SH"] == 0.0  # first of two -> not yet
    assert w.iloc[253]["SH"] == 1.0  # second consecutive -> flip


# (e) CTA backtest/live agree: with the same price panel, the CTA proxy sleeve's
# active set matches decompose_target_to_tickers' CTA ticker selection.
def test_cta_backtest_live_parity():
    from live.portfolio import (
        _cta_weights, _cta_proxy_sleeve, decompose_target_to_tickers,
    )

    n = 260
    prices = _flat_panel(n, SLEEVE_TICKERS).copy()
    # PDBC & DBMF trend up (above SMA ~100) for the last 10 days; KMLM stays flat.
    prices.iloc[250:, prices.columns.get_loc("PDBC")] = 110.0
    prices.iloc[250:, prices.columns.get_loc("DBMF")] = 105.0
    # KMLM stays at 100 (== SMA -> not strictly >, so not selected).

    rets = prices.pct_change(fill_method=None)
    w = _cta_weights(prices)
    today = prices.index[-1]
    prev = prices.index[-2]

    # Live decompose at `today`: cta budget split among today's selected set.
    target = pd.Series(
        {"A": 0.0, "B": 0.0, "rates": 0.0, "bear": 0.0, "cta": 1.0}
    )
    out = decompose_target_to_tickers(
        target, pd.Series(dtype=float), pd.Series(dtype=float), prices
    )
    selected_live = sorted(
        t for t in ["PDBC", "DBMF", "KMLM", "BIL"] if out.get(t, 0.0) > 0
    )
    selected_weights_today = sorted(
        t for t in w.columns if w.loc[today, t] > 0
    )
    assert selected_live == selected_weights_today
    assert selected_live == ["DBMF", "PDBC"]  # exactly two selected

    # Backtest sleeve return at `today` uses w[today-1] (shift(1)); same set.
    selected_backtest = sorted(
        t for t in w.columns if w.loc[prev, t] > 0
    )
    assert selected_backtest == selected_live

    # And the return equals the equal-weight mean of the selected proxies' returns
    # (NOT divided by len(available)=3) -- this is the CTA parity fix.
    expected_ret = sum(
        rets.loc[today, t] for t in selected_backtest
    ) / len(selected_backtest)
    cta_ret = _cta_proxy_sleeve(prices, rets)
    assert abs(cta_ret.loc[today] - expected_ret) < 1e-9


# (e-empty) CTA falls back to BIL when no proxy is selected.
def test_cta_bil_fallback_when_none_selected():
    from live.portfolio import _cta_weights, _cta_proxy_sleeve

    n = 260
    prices = _flat_panel(n, SLEEVE_TICKERS)  # all flat -> nothing > SMA -> none selected
    rets = prices.pct_change(fill_method=None)
    w = _cta_weights(prices)
    today = prices.index[-1]
    assert w.loc[today, "BIL"] == 1.0
    # backtest sleeve earns the BIL return on that day (0 here).
    cta_ret = _cta_proxy_sleeve(prices, rets)
    assert abs(cta_ret.loc[today] - rets.loc[today, "BIL"]) < 1e-9


# (f) sleeve cost: turnover * cost_bps/1e4 reduces net vs gross by exactly that.
def test_sleeve_cost_subtracts_known_turnover():
    from live.portfolio import build_sleeve_returns, _bear_weights

    n = 260
    prices = _flat_panel(n, SLEEVE_TICKERS).copy()
    # SPY flat at 100 for 240 days, then 95 onward -> one regime flip to bear
    # after 2 consecutive below closes (day 241). SH & BIL flat -> gross 0.
    prices.loc[:, "SPY"] = 100.0
    prices.iloc[240:, prices.columns.get_loc("SPY")] = 95.0
    prices.loc[:, "SH"] = 100.0
    prices.loc[:, "BIL"] = 100.0

    # Hand-computed bear turnover on the shifted allocation:
    #   day 1: establish BIL position        -> |w[0]| = 1.0
    #   day 242: flip BIL(1)->SH(1), BIL 1->0 -> 2.0
    #   total = 3.0
    w = _bear_weights(prices)
    w_use = w.shift(1).fillna(0.0)
    expected_turnover = float(w_use.diff().abs().sum(axis=1).fillna(0.0).sum())
    assert abs(expected_turnover - 3.0) < 1e-9

    gross = build_sleeve_returns(prices, cost_bps=0.0)["bear"]
    net = build_sleeve_returns(prices, cost_bps=10.0)["bear"]
    # Gross is 0 (flat SH/BIL); net is -turnover * cost_rate.
    assert abs(gross.sum() - 0.0) < 1e-9
    assert abs((gross - net).sum() - expected_turnover * 10.0 / 1e4) < 1e-9
    assert abs(net.sum() - (-expected_turnover * 10.0 / 1e4)) < 1e-9


# (f-per-day) cost on the specific flip day equals that day's turnover * cost_rate.
def test_sleeve_cost_per_flip_day():
    from live.portfolio import build_sleeve_returns, _bear_weights

    n = 260
    prices = _flat_panel(n, SLEEVE_TICKERS).copy()
    prices.loc[:, "SPY"] = 100.0
    prices.iloc[240:, prices.columns.get_loc("SPY")] = 95.0
    prices.loc[:, "SH"] = 100.0
    prices.loc[:, "BIL"] = 100.0

    gross = build_sleeve_returns(prices, cost_bps=0.0)["bear"]
    net = build_sleeve_returns(prices, cost_bps=10.0)["bear"]
    # On the flip day (shifted: day 242) turnover = 2.0 -> cost = 2 * 10/1e4.
    flip_day = prices.index[242]
    assert abs(
        (gross.loc[flip_day] - net.loc[flip_day]) - 2.0 * 10.0 / 1e4
    ) < 1e-9


# Shared-helper contract: backtest path shifts by 1, live path uses as-is.
def test_shared_gate_helper_timing_difference():
    from live.portfolio import _bear_weights, _bear_sleeve

    n = 260
    prices = _flat_panel(n, ["SPY", "SH", "BIL"]).copy()
    prices.loc[:, "SPY"] = 100.0
    prices.iloc[240:, prices.columns.get_loc("SPY")] = 95.0
    prices.loc[:, "SH"] = 100.0
    prices.iloc[242, prices.columns.get_loc("SH")] = 110.0
    prices.loc[:, "BIL"] = 100.0
    rets = prices.pct_change(fill_method=None)
    w = _bear_weights(prices)
    sleeve = _bear_sleeve(prices, rets)
    # Backtest sleeve at day T == (w[T-1] * rets[T]).sum -- shift(1) applied.
    for T in (241, 242, 243):
        expected = (w.shift(1).iloc[T] * rets.iloc[T]).sum()
        assert abs(sleeve.iloc[T] - expected) < 1e-9