"""Point-in-time live signal engine for Strategy A and Strategy B.

Both strategies are produced by the same dual-momentum engine; they differ only in
their factor weights and top-N concentration:

  - Strategy A (Phase 3 mom_corr): mom 0.60, corr 0.40, top_n=5.
  - Strategy B (Top-3 Dual-Momentum): mom 1.00, corr 0.00, top_n=3.

The engine mirrors the research code in ``_rsi_rotation_phase3_dual_momentum.py``
and ``_rsi_rotation_phase3_concentration_leverage.py`` exactly, but exposes a
single reusable function that returns daily net returns and target weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CoreConfig:
    """Configuration for a core dual-momentum strategy."""

    name: str
    mom_weight: float
    corr_weight: float
    top_n: int
    risk_overlay: bool = True
    use_slow_filter: bool = True


# Strategy definitions aligned with the research CSVs.
STRATEGY_A = CoreConfig(
    name="Phase3_mom_corr_A", mom_weight=0.60, corr_weight=0.40, top_n=5
)
STRATEGY_B = CoreConfig(
    name="Top3_dual_momentum_B", mom_weight=1.00, corr_weight=0.00, top_n=3
)

RSI_WINDOW = 14
MIN_HISTORY = 252
RSI_THRESHOLD = 50.0
VOL_LOOKBACK = 63
CORR_LOOKBACK = 63
VIX_PERCENTILE_LOOKBACK = 252
VIX_OVERLAY_PCTILE = 0.70

EMA_FAST = 8
EMA_MED = 21
EMA_SLOW = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "VTI", "VXUS",
    "XLK", "XLV", "XLI", "XLF", "XLE",
    "XLU", "XLP", "XLY", "XLB", "XLRE",
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
    "META", "TSLA", "JPM",
    # ponytail: BTC-USD/ETH-USD dropped — not Alpaca-equity-orderable and trade 7
    # days/week (calendar risk on an equity-session panel). Re-enable later as its
    # own sleeve with correct Alpaca crypto symbols if crypto exposure is wanted.
]
CASH_PROXY = "BIL"
VIX_TICKER = "^VIX"


def _rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # ponytail: set the zero-loss/zero-gain boundaries exactly instead of NaN.
    # avg_loss==0 -> RSI 100 (gains, no losses); avg_gain==0 with losses -> RSI 0;
    # flat (neither) -> 50. The old .replace(0, NaN) made the zero-loss case NaN,
    # which then failed `momentum > RSI_THRESHOLD` and dropped the asset.
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _ensemble_signals(close_panel: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    signals = pd.DataFrame(0, index=close_panel.index, columns=tickers, dtype=np.int8)
    for t in tickers:
        if t not in close_panel.columns:
            continue
        close = close_panel[t]
        e_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
        e_med = close.ewm(span=EMA_MED, adjust=False).mean()
        e_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
        ema_long = (e_fast > e_med) & (e_med > e_slow)

        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow_macd = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow_macd
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        macd_long = macd_line > signal_line

        signals[t] = (ema_long & macd_long).astype(int)
    return signals


def _rebalance_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(dates.weekday == 4)


def _adaptive_momentum(close_panel: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    available_universe = [t for t in UNIVERSE if t in close_panel.columns]

    rsi_layers: Dict[int, pd.DataFrame] = {}
    for lb in [63, 126, 252]:
        cum = close_panel[available_universe] / close_panel[available_universe].shift(lb)
        rsi_layers[lb] = cum.apply(lambda col: _rsi(col, RSI_WINDOW))

    vix_pct = vix.rolling(VIX_PERCENTILE_LOOKBACK, min_periods=126).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1], kind="rank") / 100.0, raw=False
    )

    w63 = 0.50 + 0.50 * vix_pct.apply(
        lambda x: 0.0 if pd.isna(x) else (max(0.0, min(x - 0.67, 0.33)) / 0.33)
    )
    w252 = 0.50 * vix_pct.apply(
        lambda x: 0.0 if pd.isna(x) else (max(0.0, min(0.33 - x, 0.33)) / 0.33)
    )
    w126 = 1.0 - w63 - w252

    blended = pd.DataFrame(0.0, index=close_panel.index, columns=available_universe)
    for t in available_universe:
        blended[t] = w63 * rsi_layers[63][t] + w126 * rsi_layers[126][t] + w252 * rsi_layers[252][t]
    return blended


def _slow_trend_filter(close_panel: pd.DataFrame) -> pd.DataFrame:
    available_universe = [t for t in UNIVERSE if t in close_panel.columns]
    sma252 = close_panel[available_universe].rolling(252, min_periods=126).mean()
    return (close_panel[available_universe] > sma252).astype(int)


def _avg_offdiag_corr(corr_mat: pd.DataFrame) -> pd.Series:
    """Per-asset average correlation EXCLUDING the self-diagonal (==1.0).

    ``corr_mat`` is a rolling-corr frame with a (date, ticker) MultiIndex and ticker
    columns. Sum and count both include the diagonal, so subtract 1 from each; the
    count-1 denominator is the number of off-diagonal pairs and handles NaN pairs.
    """
    offdiag_sum = corr_mat.groupby(level=0).sum() - 1.0
    offdiag_count = corr_mat.groupby(level=0).count() - 1
    return offdiag_sum / offdiag_count.replace(0, np.nan)


def _factor_scores(
    close_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    ensemble: pd.DataFrame,
    enough_history: pd.DataFrame,
    momentum: pd.DataFrame,
    mom_weight: float,
    corr_weight: float,
) -> pd.DataFrame:
    available_universe = [t for t in UNIVERSE if t in close_panel.columns]
    vol = returns_panel[available_universe].rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK // 2).std() * np.sqrt(252)
    vol_score = 1.0 / vol.replace(0, np.nan)

    corr_mat = returns_panel[available_universe].rolling(CORR_LOOKBACK, min_periods=CORR_LOOKBACK // 2).corr()
    # ponytail: exclude the self-correlation diagonal (==1.0) from the per-asset
    # average — the old groupby(level=0).mean() added a guaranteed 1.0 to every
    # score.
    avg_corr = _avg_offdiag_corr(corr_mat)
    corr_score = 1.0 - avg_corr

    def rank01(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    composite = mom_weight * rank01(momentum) + corr_weight * rank01(corr_score)
    if (mom_weight + corr_weight) > 0:
        composite = composite / (mom_weight + corr_weight)

    gate = (momentum > RSI_THRESHOLD) & enough_history & (ensemble == 1)
    return composite.where(gate)


def _run_core_engine(
    close_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    cash_return: pd.Series,
    benchmark_return: pd.Series,
    vix: pd.Series,
    config: CoreConfig,
    commission_bps: float,
) -> pd.DataFrame:
    dates = close_panel.index
    available_universe = [t for t in UNIVERSE if t in close_panel.columns]

    rebalance = _rebalance_mask(dates)
    momentum = _adaptive_momentum(close_panel, vix)
    ensemble = _ensemble_signals(close_panel, available_universe)
    enough_history = close_panel[available_universe].notna().cumsum() >= MIN_HISTORY
    slow_filter = _slow_trend_filter(close_panel) if config.use_slow_filter else pd.DataFrame(
        1, index=dates, columns=available_universe
    )
    composite = _factor_scores(
        close_panel, returns_panel, ensemble, enough_history, momentum,
        config.mom_weight, config.corr_weight,
    )

    target_weights = pd.DataFrame(0.0, index=dates, columns=available_universe)
    prev_weights = pd.Series(0.0, index=available_universe)

    for i in range(1, len(dates)):
        today = dates[i]
        if not rebalance[i]:
            target_weights.loc[today] = prev_weights.values
            continue

        yesterday = dates[i - 1]
        scores_y = composite.loc[yesterday].where(slow_filter.loc[yesterday] == 1)
        top = scores_y.dropna().sort_values(ascending=False).head(config.top_n)

        day_weights = pd.Series(0.0, index=available_universe)
        if len(top) > 0:
            w = 1.0 / config.top_n
            for t in top.index:
                day_weights[t] = w

        if config.risk_overlay:
            vix_pct = stats.percentileofscore(
                vix.iloc[max(0, i - 252):i].dropna(), vix.iloc[i - 1]
            ) / 100.0
            if vix_pct > VIX_OVERLAY_PCTILE:
                day_weights = day_weights * 0.5

        target_weights.loc[today] = day_weights.values
        prev_weights = day_weights

    gross_return = (target_weights * returns_panel[available_universe]).sum(axis=1)
    turnover = target_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = target_weights.iloc[0].abs().sum()
    cost = turnover * (commission_bps / 10000.0)
    net_return = gross_return - cost
    cash_weight = 1.0 - target_weights.sum(axis=1)
    net_return = net_return + cash_weight.clip(lower=0.0) * cash_return

    result = pd.DataFrame({
        "strategy_return": net_return,
        "benchmark_return": benchmark_return,
        "cash_weight": cash_weight,
        "turnover": turnover,
        "cost": cost,
    }, index=dates)
    for ticker in available_universe:
        result[f"weight_{ticker}"] = target_weights[ticker]
    result["cumulative_return"] = (1 + result["strategy_return"].fillna(0)).cumprod()
    return result.dropna(subset=["strategy_return"])


def build_core_returns(
    close_panel: pd.DataFrame,
    commission_bps: float = 10.0,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Compute live, point-in-time net returns for Strategy A and Strategy B.

    Parameters
    ----------
    close_panel : pd.DataFrame
        Adjusted closes for UNIVERSE + VIX + cash proxy, indexed by date.
    commission_bps : float
        One-way transaction cost in basis points.

    Returns
    -------
    ret_a : pd.Series
        Daily net returns for Strategy A.
    ret_b : pd.Series
        Daily net returns for Strategy B.
    weights_a : pd.DataFrame
        Daily target weights for Strategy A.
    weights_b : pd.DataFrame
        Daily target weights for Strategy B.
    """
    all_dates = close_panel.index
    returns_panel = close_panel.pct_change(fill_method=None)
    cash_return = returns_panel.get(CASH_PROXY, pd.Series(0.0, index=all_dates))

    available_universe = [t for t in UNIVERSE if t in close_panel.columns]
    benchmark_return = returns_panel[available_universe].mean(axis=1, skipna=True)

    vix_col = VIX_TICKER if VIX_TICKER in close_panel.columns else "VIX"
    vix = close_panel[vix_col]

    res_a = _run_core_engine(
        close_panel, returns_panel, cash_return, benchmark_return, vix,
        STRATEGY_A, commission_bps,
    )
    res_b = _run_core_engine(
        close_panel, returns_panel, cash_return, benchmark_return, vix,
        STRATEGY_B, commission_bps,
    )

    weight_cols = [c for c in res_a.columns if c.startswith("weight_")]
    return (
        res_a["strategy_return"],
        res_b["strategy_return"],
        res_a[weight_cols],
        res_b[weight_cols],
    )


if __name__ == "__main__":  # ponytail: one runnable check for the RSI boundary fix.
    import numpy as np

    rng = np.arange(100.0, 100.0 + RSI_WINDOW + 5)  # strictly rising -> no losses
    up = pd.Series(rng)
    rsi_up = float(_rsi(up).iloc[-1])
    assert rsi_up == 100.0, f"monotonic-up RSI should be 100, got {rsi_up}"

    down = pd.Series(-rng)  # strictly falling -> no gains
    rsi_down = float(_rsi(down).iloc[-1])
    assert rsi_down == 0.0, f"monotonic-down RSI should be 0, got {rsi_down}"

    flat = pd.Series(np.full(RSI_WINDOW + 5, 100.0))  # no moves
    rsi_flat = float(_rsi(flat).iloc[-1])
    assert rsi_flat == 50.0, f"flat RSI should be 50, got {rsi_flat}"

    print(f"core_signals self-check OK: RSI up={rsi_up} down={rsi_down} flat={rsi_flat}")
