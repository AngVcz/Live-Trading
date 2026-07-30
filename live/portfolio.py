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


def _rates_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    available = [t for t in ["TLT", "IEF", "BIL"] if t in prices.columns]
    if len(available) < 3:
        raise ValueError(f"Rates sleeve missing tickers: {available}")
    sma = prices[["TLT", "IEF", "BIL"]].rolling(TREND_WINDOW, min_periods=126).mean()
    long_tlt = prices["TLT"] > sma["TLT"]
    long_ief = prices["IEF"] > sma["IEF"]
    out = pd.Series(0.0, index=rets.index)
    out.loc[long_tlt] = rets.loc[long_tlt, "TLT"]
    out.loc[~long_tlt & long_ief] = rets.loc[~long_tlt & long_ief, "IEF"]
    out.loc[~(long_tlt | long_ief)] = rets.loc[~(long_tlt | long_ief), "BIL"]
    return out


def _bear_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    available = [t for t in ["SPY", "SH", "BIL"] if t in prices.columns]
    if "SPY" not in available:
        raise ValueError("Bear sleeve requires SPY")
    sma = prices["SPY"].rolling(TREND_WINDOW, min_periods=126).mean()
    in_bear = prices["SPY"] < sma
    out = pd.Series(0.0, index=rets.index)
    if "SH" in available:
        out.loc[in_bear] = rets.loc[in_bear, "SH"]
    out.loc[~in_bear] = rets.loc[~in_bear, "BIL"]
    return out


def _cta_proxy_sleeve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    available = [t for t in ["PDBC", "DBMF", "KMLM"] if t in prices.columns]
    if not available:
        return pd.Series(0.0, index=rets.index)
    sma = prices[available].rolling(TREND_WINDOW, min_periods=126).mean()
    long_flags = prices[available] > sma
    out = pd.Series(0.0, index=rets.index)
    for t in available:
        out.loc[long_flags[t]] += rets.loc[long_flags[t], t] / len(available)
    return out


def build_sleeve_returns(
    prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute daily returns for the three diversifier sleeves.

    Returns
    -------
    pd.DataFrame with columns rates, bear, cta indexed by date.
    """
    tickers = list(tickers or SLEEVE_TICKERS)
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available].copy()
    rets = prices.pct_change(fill_method=None)

    return pd.DataFrame({
        "rates": _rates_sleeve(prices, rets),
        "bear": _bear_sleeve(prices, rets),
        "cta": _cta_proxy_sleeve(prices, rets),
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
        Ticker -> target weight as fraction of NAV.
    """
    out: Dict[str, float] = {}

    # Normalize A/B internal weights to allocate their full sleeve budget.
    weight_a = weight_a.copy()
    weight_b = weight_b.copy()
    if weight_a.abs().sum() > 0:
        weight_a = weight_a / weight_a.abs().sum()
    if weight_b.abs().sum() > 0:
        weight_b = weight_b / weight_b.abs().sum()

    for ticker, w in weight_a.items():
        clean = ticker.replace("weight_", "")
        out[clean] = out.get(clean, 0.0) + float(target_weights.loc["A"]) * w

    for ticker, w in weight_b.items():
        clean = ticker.replace("weight_", "")
        out[clean] = out.get(clean, 0.0) + float(target_weights.loc["B"]) * w

    today = prices.index[-1]
    sma_spy = prices["SPY"].rolling(TREND_WINDOW, min_periods=126).mean()

    # Rates sleeve: pick TLT / IEF / BIL based on 200-day trend.
    rates_today = "BIL"
    if "TLT" in prices.columns and prices.loc[today, "TLT"] > prices["TLT"].rolling(TREND_WINDOW, min_periods=126).mean().loc[today]:
        rates_today = "TLT"
    elif "IEF" in prices.columns and prices.loc[today, "IEF"] > prices["IEF"].rolling(TREND_WINDOW, min_periods=126).mean().loc[today]:
        rates_today = "IEF"
    out[rates_today] = out.get(rates_today, 0.0) + float(target_weights.loc["rates"])

    # Bear sleeve: SH when SPY is below SMA200, otherwise BIL.
    in_bear = prices.loc[today, "SPY"] < sma_spy.loc[today]
    bear_today = "SH" if in_bear and "SH" in prices.columns else "BIL"
    out[bear_today] = out.get(bear_today, 0.0) + float(target_weights.loc["bear"])

    # CTA sleeve: equal-weight among trend-up proxies, else BIL.
    cta_tickers = [t for t in ["PDBC", "DBMF", "KMLM"] if t in prices.columns]
    cta_sma = {t: prices[t].rolling(TREND_WINDOW, min_periods=126).mean().loc[today] for t in cta_tickers}
    cta_selected = [t for t in cta_tickers if prices.loc[today, t] > cta_sma[t]]
    if not cta_selected:
        cta_selected = ["BIL"]
    for t in cta_selected:
        out[t] = out.get(t, 0.0) + float(target_weights.loc["cta"]) / len(cta_selected)

    return out
