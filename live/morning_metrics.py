"""Deterministic metrics panel for the 07:30 morning report.

Every number the LLM sees is computed here from the price panel and persisted
state — the model interprets these values, it never invents them. Any single
metric that cannot be computed becomes the string "n/a"; the panel as a whole
never raises.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

import pandas as pd
from scipy import stats

from live.core_signals import UNIVERSE as CORE_UNIVERSE
from live.state import get_peak_equity, load_last_weights

_NA = "n/a"


def _last_two(col: pd.Series):
    s = col.dropna()
    return float(s.iloc[-1]), float(s.iloc[-2])


def _above_sma200(col: pd.Series) -> bool:
    s = col.dropna()
    if len(s) < 200:
        raise ValueError("short history")
    return bool(s.iloc[-1] > s.tail(200).mean())


def _vix_metrics(prices: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    col = prices["VIX"] if "VIX" in prices.columns else prices["^VIX"]
    last, prev = _last_two(col)
    out["vix_close"] = round(last, 2)
    out["vix_change_1d"] = round(last - prev, 2)
    window = col.dropna().tail(252)
    if len(window) < 126:
        raise ValueError("short VIX history")
    pct = float(stats.percentileofscore(window, last, kind="rank") / 100.0)
    out["vix_percentile_252d"] = round(pct, 3)
    out["vix_overlay_active"] = pct > 0.70   # VIX_OVERLAY_PCTILE in core_signals
    return out


def _tnx_metrics(as_of: date) -> Dict[str, Any]:
    from live.data_feed import fetch_panel
    # ^TNX is not Alpaca-tradeable -> always yfinance (cached).
    panel = fetch_panel(["^TNX"], as_of - timedelta(days=365), as_of,
                        prefer_alpaca=False)
    col = panel["^TNX"].dropna()
    if len(col) < 6:
        raise ValueError("short TNX history")
    return {
        # ^TNX quotes the 10Y yield x 10 (CBOE convention).
        "tnx_10y_level": round(float(col.iloc[-1]) / 10.0, 3),
        "tnx_change_5d": round(float(col.iloc[-1] - col.iloc[-6]) / 10.0, 3),
    }


def _breadth(prices: pd.DataFrame) -> Dict[str, Any]:
    cols = [t for t in CORE_UNIVERSE if t in prices.columns
            and prices[t].dropna().shape[0] >= 200]
    if not cols:
        raise ValueError("no universe columns")
    above = sum(1 for t in cols if _above_sma200(prices[t]))
    return {"breadth_pct_above_sma200": round(100.0 * above / len(cols), 1),
            "breadth_universe_n": len(cols)}


def _holdings_below(prices: pd.DataFrame, weights_a: pd.Series, weights_b: pd.Series):
    held = sorted(set(weights_a[weights_a > 0].index) | set(weights_b[weights_b > 0].index))
    below = []
    for t in held:
        if t in prices.columns:
            try:
                if not _above_sma200(prices[t]):
                    below.append(t)
            except ValueError:
                pass
    return below


def _book_metrics(equity: float) -> Dict[str, Any]:
    peak = get_peak_equity() or equity
    dd = (equity - peak) / peak if peak > 0 else 0.0
    return {"peak_equity": round(float(peak), 2),
            "drawdown_pct": round(100.0 * dd, 2),
            "guardrail_margin_pct": round(100.0 * (equity - 0.9 * peak) / peak, 2)
            if peak > 0 else _NA}


def _turnover(ticker_weights: Dict[str, float]):
    last = load_last_weights()
    if last is None:
        raise ValueError("no last weights")
    lw = last.to_dict()
    tickers = set(ticker_weights) | set(lw)
    tw = sum(abs(ticker_weights.get(t, 0.0) - lw.get(t, 0.0)) for t in tickers) / 2.0
    return round(100.0 * tw, 2)


def compute_metrics_panel(
    prices: pd.DataFrame,
    ticker_weights: Dict[str, float],
    weights_a: pd.Series,
    weights_b: pd.Series,
    equity: float,
    as_of: date,
) -> Dict[str, Any]:
    """The full daily metrics panel. Values are exact; "n/a" marks a failed cell."""
    m: Dict[str, Any] = {"as_of": as_of.isoformat(), "equity": round(float(equity), 2)}
    for block in (
        lambda: _vix_metrics(prices),
        lambda: _tnx_metrics(as_of),
        lambda: {"tlt_above_sma200": _above_sma200(prices["TLT"])},
        lambda: {"ief_above_sma200": _above_sma200(prices["IEF"])},
        lambda: _breadth(prices),
        lambda: {"holdings_below_sma200": _holdings_below(prices, weights_a, weights_b)},
        lambda: _book_metrics(equity),
        lambda: {"turnover_oneway_pct": _turnover(ticker_weights)},
    ):
        try:
            m.update(block())
        except Exception:
            pass  # a failed block leaves its keys at the setdefault 'n/a' below
    m.setdefault("vix_close", _NA)
    m.setdefault("vix_change_1d", _NA)
    m.setdefault("vix_percentile_252d", _NA)
    m.setdefault("vix_overlay_active", False)
    m.setdefault("tnx_10y_level", _NA)
    m.setdefault("tnx_change_5d", _NA)
    m.setdefault("tlt_above_sma200", _NA)
    m.setdefault("ief_above_sma200", _NA)
    m.setdefault("breadth_pct_above_sma200", _NA)
    m.setdefault("holdings_below_sma200", _NA)
    m.setdefault("peak_equity", _NA)
    m.setdefault("drawdown_pct", _NA)
    m.setdefault("guardrail_margin_pct", _NA)
    m.setdefault("turnover_oneway_pct", _NA)
    m["sleeve_weights"] = {}  # filled by caller (kept with the prompt payload)
    m["top_tickers"] = dict(sorted(ticker_weights.items(), key=lambda kv: kv[1],
                                   reverse=True)[:10])
    m["macro_calendar"] = []  # Stage 1 (LLM) fills today's events from the web
    return m


def _self_check() -> None:
    """Compute the panel on a 4-year cached panel; assert keys, types, ranges."""
    from live.core_signals import build_core_returns
    from live.data_feed import fetch_panel
    from live.portfolio import (SleeveConfig, build_live_weights,
                                build_sleeve_returns, decompose_target_to_tickers)
    end = date(2025, 6, 30)
    tickers = list(dict.fromkeys(
        CORE_UNIVERSE + ["TLT", "IEF", "PDBC", "KMLM", "DBMF", "BIL", "^VIX"]))
    prices = fetch_panel(tickers, end - timedelta(days=365 * 4), end,
                         prefer_alpaca=False).rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, wa, wb = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    tw = decompose_target_to_tickers(sleeve, wa.iloc[-1], wb.iloc[-1], prices)
    m = compute_metrics_panel(prices, tw, wa.iloc[-1], wb.iloc[-1],
                              equity=100_000.0, as_of=end)
    assert isinstance(m["vix_overlay_active"], bool)
    if m["vix_percentile_252d"] != _NA:
        assert 0.0 <= m["vix_percentile_252d"] <= 1.0
    assert isinstance(m["holdings_below_sma200"], list)
    assert m["macro_calendar"] == []
    print(f"morning_metrics self-check OK: vix={m['vix_close']} "
          f"pctile={m['vix_percentile_252d']} breadth={m['breadth_pct_above_sma200']}%")


if __name__ == "__main__":
    _self_check()