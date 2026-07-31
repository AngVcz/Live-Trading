"""Portfolio construction logic for the live A+B+Diversifier Sleeves strategy.

Re-implements the sleeve-building rules from ``EnsembleABSleevesStrategy`` in a
standalone, testable form so the live runner can compute target dollar allocations
without depending on the Streamlit dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SleeveConfig:
    """Target mix for the live portfolio."""

    weight_a: float = 0.20
    weight_b: float = 0.20
    weight_rates: float = 0.20
    weight_bear: float = 0.20
    weight_cta: float = 0.20
    rebalance_freq: str = "Y"          # 'Y' = annual, 'M' = monthly, 'W' = weekly
    drift_threshold: float = 0.10        # absolute deviation that triggers rebalance
    commission_bps: float = 10.0

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "A": self.weight_a,
            "B": self.weight_b,
            "rates": self.weight_rates,
            "bear": self.weight_bear,
            "cta": self.weight_cta,
        }


SLEEVE_TICKERS = ["SPY", "TLT", "IEF", "GLD", "PDBC", "KMLM", "DBMF", "VIXM", "SH", "BIL", "VIXY", "PSQ"]
TREND_WINDOW = 200
# ponytail: N=2 consecutive closes beyond SMA200 before a regime flip (whipsaw guard).
HYSTERESIS_N = 2


def _hysteresis_regime(condition: pd.Series, n: int = HYSTERESIS_N) -> pd.Series:
    """Boolean regime flag that only flips after ``n`` consecutive closes against
    the current regime (whipsaw filter).

    ``condition`` True = the 'on' state (e.g. price > SMA = uptrend). The flag is
    the single shared gate used by BOTH the backtest sleeve path and the live
    ``decompose_target_to_tickers`` path, so the two cannot drift.

    Timing: the helper produces the regime flag *on a date* from that date's
    close. The backtest sleeve path shifts it by 1 (decide on close[T-1], earn
    close[T-1] -> close[T]); the live decompose path uses it as-is for today's
    target (the human allocates after the close).

    ponytail: iterative scan -- O(T) and simple; series are short (daily history).
    """
    vals = np.asarray(condition.fillna(False).astype(bool))
    out = np.zeros(len(vals), dtype=bool)
    state = False
    streak = 0          # consecutive observations opposite to `state`
    seeded = False
    for i, v in enumerate(vals):
        v = bool(v)
        if not seeded:
            state = v
            streak = 0
            out[i] = state
            seeded = True
            continue
        if v == state:
            streak = 0
            out[i] = state
        else:
            streak += 1
            if streak >= n:
                state = v
                streak = 0
            out[i] = state
    return pd.Series(out, index=condition.index)


def _rates_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-constituent allocation weights for the rates sleeve (hysteresis'd).

    Exactly one of TLT / IEF / BIL holds 1.0 each day (TLT wins over IEF).
    """
    available = [t for t in ["TLT", "IEF", "BIL"] if t in prices.columns]
    if len(available) < 3:
        raise ValueError(f"Rates sleeve missing tickers: {available}")
    sma = prices[["TLT", "IEF", "BIL"]].rolling(TREND_WINDOW, min_periods=126).mean()
    long_tlt = _hysteresis_regime(prices["TLT"] > sma["TLT"])
    long_ief = _hysteresis_regime(prices["IEF"] > sma["IEF"])
    w = pd.DataFrame(0.0, index=prices.index, columns=["TLT", "IEF", "BIL"])
    w["TLT"] = long_tlt.astype(float)
    w["IEF"] = (~long_tlt & long_ief).astype(float)
    w["BIL"] = (~long_tlt & ~long_ief).astype(float)
    return w


def _bear_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-constituent allocation weights for the bear sleeve (hysteresis'd)."""
    available = [t for t in ["SPY", "SH", "BIL"] if t in prices.columns]
    if "SPY" not in available:
        raise ValueError("Bear sleeve requires SPY")
    sma = prices["SPY"].rolling(TREND_WINDOW, min_periods=126).mean()
    in_bear = _hysteresis_regime(prices["SPY"] < sma)
    w = pd.DataFrame(0.0, index=prices.index, columns=["SH", "BIL"])
    if "SH" in available:
        w["SH"] = in_bear.astype(float)
    w["BIL"] = 1.0 - w["SH"]
    return w


def _cta_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-constituent allocation weights for the CTA proxy sleeve (hysteresis'd).

    Equal-weight among the selected (in-uptrend) proxies; falls back to BIL when
    none are selected or no proxies are available. Each row sums to 1.0.
    """
    available = [t for t in ["PDBC", "DBMF", "KMLM"] if t in prices.columns]
    cols = available + (["BIL"] if "BIL" in prices.columns else [])
    w = pd.DataFrame(0.0, index=prices.index, columns=cols)
    if not available:
        if "BIL" in prices.columns:
            w["BIL"] = 1.0
        return w
    sma = prices[available].rolling(TREND_WINDOW, min_periods=126).mean()
    flags = pd.DataFrame(
        {t: _hysteresis_regime(prices[t] > sma[t]) for t in available}
    )
    n_selected = flags.sum(axis=1)
    # ponytail: divide by selected count (not len(available)); floor to BIL when 0.
    denom = n_selected.where(n_selected > 0, 1.0)
    for t in available:
        w[t] = flags[t].astype(float) / denom
    if "BIL" in prices.columns:
        w["BIL"] = (n_selected == 0).astype(float)
    return w


def _net_sleeve_return(
    w_df: pd.DataFrame, rets: pd.DataFrame, cost_rate: float
) -> pd.Series:
    """Net sleeve return = gross(shifted allocation) - turnover * cost_rate.

    Decision is made on close[T-1] and earns close[T-1] -> close[T], so the
    allocation is shifted by 1 before multiplying returns. Turnover is the L1
    change in the sleeve's allocation; the cost is charged to the return that
    the trade enables. ``cost_rate == 0`` yields the gross return.
    """
    cols = [c for c in w_df.columns if c in rets.columns]
    if not cols:
        return pd.Series(0.0, index=rets.index)
    # decide on T-1; no position held before the first decision
    w = w_df[cols].shift(1).fillna(0.0)
    gross = (w * rets[cols]).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * cost_rate


def _rates_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Gross rates sleeve return (no cost); decision on T-1."""
    return _net_sleeve_return(_rates_weights(prices), rets, 0.0)


def _bear_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Gross bear sleeve return (no cost); decision on T-1."""
    return _net_sleeve_return(_bear_weights(prices), rets, 0.0)


def _cta_proxy_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Gross CTA proxy sleeve return (no cost); decision on T-1.

    Equal-weight among selected proxies, BIL fallback -- matches
    ``decompose_target_to_tickers``.
    """
    return _net_sleeve_return(_cta_weights(prices), rets, 0.0)


def build_sleeve_returns(
    prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Compute daily returns for the three diversifier sleeves.

    Returns
    -------
    pd.DataFrame with columns rates, bear, cta indexed by date. Each sleeve
    return is net of a turnover-based transaction cost (``cost_bps`` bps per
    unit of allocation turnover).
    """
    tickers = list(tickers or SLEEVE_TICKERS)
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available].copy()
    rets = prices.pct_change(fill_method=None)

    cost_rate = cost_bps / 1e4
    return pd.DataFrame({
        "rates": _net_sleeve_return(_rates_weights(prices), rets, cost_rate),
        "bear": _net_sleeve_return(_bear_weights(prices), rets, cost_rate),
        "cta": _net_sleeve_return(_cta_weights(prices), rets, cost_rate),
    }, index=prices.index)


def build_live_weights(
    ret_a: pd.Series,
    ret_b: pd.Series,
    sleeve_rets: pd.DataFrame,
    config: SleeveConfig,
    last_weights: Optional[pd.Series] = None,
    today: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Compute target weights for today using the fixed sleeve allocation.

    Parameters
    ----------
    ret_a, ret_b : pd.Series
        Historical daily net returns for core strategies A and B.
    sleeve_rets : pd.DataFrame
        Historical daily returns for rates/bear/cta sleeves.
    config : SleeveConfig
    last_weights : pd.Series, optional
        Weights from the previous rebalance. Used to check drift.
    today : pd.Timestamp, optional
        Date for which weights are computed; defaults to last available.

    Returns
    -------
    pd.Series
        Mapping ticker -> target weight.
    """
    common_index = ret_a.index.intersection(ret_b.index).intersection(sleeve_rets.index)
    if today is None:
        today = common_index[-1]

    # Determine if this is a rebalance date.
    is_rebalance = False
    if config.rebalance_freq == "Y":
        is_rebalance = (today.month == 1 and today.day <= 5)
    elif config.rebalance_freq == "M":
        is_rebalance = today.is_month_start or today.day <= 5
    elif config.rebalance_freq == "W":
        is_rebalance = today.weekday() == 4

    target = pd.Series({
        "A": config.weight_a,
        "B": config.weight_b,
        "rates": config.weight_rates,
        "bear": config.weight_bear,
        "cta": config.weight_cta,
    })

    # last_weights es un Series de tickers (persistido en state.json). Si el índice
    # no coincide con el de sleeves, ignorarlo.
    if last_weights is not None and not is_rebalance:
        if set(last_weights.index) == set(target.index):
            drift = (target - last_weights).abs().max()
            is_rebalance = drift > config.drift_threshold

    if not is_rebalance and last_weights is not None:
        if set(last_weights.index) == set(target.index):
            return last_weights.copy()

    return target


def _deploy_sleeve_internal(
    out: Dict[str, float],
    weight: pd.Series,
    budget: float,
) -> None:
    """Deploy a core sleeve's raw internal weights and route the cash residual to BIL.

    Do NOT renormalize: if the raw weights sum to < 1 (e.g. after the VIX 0.5 cut),
    the residual ``max(0, 1 - raw_sum)`` is deployed to BIL as cash ballast. When
    the sleeve is empty (raw_sum == 0) the whole budget goes to BIL. ``raw_sum``
    never exceeds 1, so ``max(0, ...)`` is safe.
    """
    weight = weight.copy()
    raw_sum = float(weight.abs().sum())
    for ticker, w in weight.items():
        clean = ticker.replace("weight_", "")
        out[clean] = out.get(clean, 0.0) + budget * float(w)
    residual = max(0.0, 1.0 - raw_sum)
    if residual > 0.0:
        out["BIL"] = out.get("BIL", 0.0) + budget * residual


def decompose_target_to_tickers(
    target_weights: pd.Series,
    weight_a: pd.Series,
    weight_b: pd.Series,
    prices: pd.DataFrame,
) -> Dict[str, float]:
    """
    Convert sleeve-level weights into individual ticker target weights.

    Parameters
    ----------
    target_weights : pd.Series
        Weights for A/B/rates/bear/cta.
    weight_a : pd.Series
        Latest target weights inside Strategy A.
    weight_b : pd.Series
        Latest target weights inside Strategy B.
    prices : pd.DataFrame
        Used only to verify ticker availability and compute sleeve signals.

    Returns
    -------
    dict[str, float]
        Ticker -> target weight as fraction of NAV. Sums to 1.0.
    """
    out: Dict[str, float] = {}

    # A/B: deploy raw internal exposure; route the unused cash to BIL (ballast).
    _deploy_sleeve_internal(out, weight_a, float(target_weights.loc["A"]))
    _deploy_sleeve_internal(out, weight_b, float(target_weights.loc["B"]))

    today = prices.index[-1]

    # Rates sleeve: today's allocation from the shared hysteresis'd gate.
    rates_w = _rates_weights(prices)
    rates_today = "BIL"
    for t in ["TLT", "IEF", "BIL"]:
        if t in rates_w.columns and rates_w.loc[today, t] > 0:
            rates_today = t
            break
    out[rates_today] = out.get(rates_today, 0.0) + float(target_weights.loc["rates"])

    # Bear sleeve: today's allocation from the shared hysteresis'd gate.
    bear_w = _bear_weights(prices)
    bear_today = "SH" if ("SH" in bear_w.columns and bear_w.loc[today, "SH"] > 0) else "BIL"
    out[bear_today] = out.get(bear_today, 0.0) + float(target_weights.loc["bear"])

    # CTA sleeve: today's allocation from the shared hysteresis'd gate (equal-weight
    # of selected proxies, BIL floor) -- identical selection to the backtest path.
    cta_w = _cta_weights(prices)
    cta_today_alloc = cta_w.loc[today]
    cta_selected = [t for t in cta_w.columns if cta_today_alloc[t] > 0]
    if not cta_selected:
        cta_selected = ["BIL"]
    for t in cta_selected:
        out[t] = out.get(t, 0.0) + float(target_weights.loc["cta"]) / len(cta_selected)

    return out