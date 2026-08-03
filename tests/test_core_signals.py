"""Unit tests for live/core_signals.py RSI boundary + off-diagonal correlation fix.

Offline and deterministic: synthetic Series/DataFrames, no network, no Date/random.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.core_signals import _rsi, _avg_offdiag_corr, RSI_WINDOW


def test_rsi_monotonic_up_is_100_not_nan():
    """A strictly rising series has gains and no losses -> RSI 100 (was NaN)."""
    up = pd.Series(np.arange(100.0, 100.0 + RSI_WINDOW + 5))
    assert float(_rsi(up).iloc[-1]) == 100.0


def test_rsi_monotonic_down_is_zero():
    """A strictly falling series has losses and no gains -> RSI 0."""
    down = pd.Series(-np.arange(100.0, 100.0 + RSI_WINDOW + 5))
    assert float(_rsi(down).iloc[-1]) == 0.0


def test_rsi_flat_is_50():
    """No gains and no losses -> RSI 50 (neutral), not NaN."""
    flat = pd.Series(np.full(RSI_WINDOW + 5, 100.0))
    assert float(_rsi(flat).iloc[-1]) == 50.0


def test_avg_offdiag_corr_excludes_diagonal():
    """The self-correlation diagonal (1.0) must NOT bias the per-asset average.

    3 assets, all pairwise off-diagonal correlations 0.5 -> avg_corr must be 0.5
    (NOT (0.5+0.5+1.0)/3 == 0.667, which the old groupby.mean() produced).
    """
    tickers = ["A", "B", "C"]
    dates = pd.to_datetime(["2025-01-01"])
    mat = np.full((3, 3), 0.5)
    np.fill_diagonal(mat, 1.0)
    corr_mat = pd.DataFrame(
        mat,
        index=pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]),
        columns=tickers,
    )
    avg = _avg_offdiag_corr(corr_mat).loc[dates[0]]
    # Each asset: two off-diagonal entries of 0.5 -> mean 0.5.
    assert np.allclose(avg.values, 0.5)


def test_avg_offdiag_corr_two_assets_identical():
    """2 perfectly-correlated assets: the single off-diagonal is 1.0 -> avg 1.0."""
    tickers = ["A", "B"]
    dates = pd.to_datetime(["2025-01-01"])
    mat = np.array([[1.0, 1.0], [1.0, 1.0]])
    corr_mat = pd.DataFrame(
        mat,
        index=pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]),
        columns=tickers,
    )
    avg = _avg_offdiag_corr(corr_mat).loc[dates[0]]
    assert np.allclose(avg.values, 1.0)