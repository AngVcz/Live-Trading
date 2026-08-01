"""Single-trial validation of the SMA200 band rates gate.

The band=2.5% config is pre-registered as the one mechanistically-motivated
candidate: it directly addresses the whipsaw around SMA200 and was on the OOS
frontier in the earlier sweep. This script tests it as ONE hypothesis on a
genuinely held-out split (train 2015-2019, test 2020-2025), reporting Sharpe,
MaxDD, turnover, PSR, and DSR with n_trials=1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from live.core_signals import build_core_returns
from live.portfolio import build_sleeve_returns, _rates_weights

PX = pd.read_parquet("cache/_full_panel_2015_2026.parquet")
COST_BPS = 10.0
BAND = 0.025

TRAIN_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan


def cagr_dd(r: pd.Series):
    r = r.dropna()
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1 if yrs else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return cagr, dd


def turns_per_year(w_df: pd.DataFrame) -> float:
    w = w_df.shift(1).fillna(0.0)
    l1 = w.diff().abs().sum(axis=1).fillna(0.0)
    return (l1.mean() * 252) / 2.0


def probabilistic_sharpe_ratio(returns: pd.Series, target_sharpe: float = 0.0) -> float:
    r = returns.dropna()
    t = len(r)
    if t < 30:
        return np.nan
    mean, std = r.mean(), r.std(ddof=1)
    if std == 0:
        return np.nan
    sr_daily = mean / std
    sr = sr_daily * np.sqrt(252)
    skew = r.skew()
    kurt = r.kurtosis() + 3.0
    denom_sq = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)
    if denom_sq <= 0:
        return np.nan
    z = (sr - target_sharpe) * np.sqrt(t - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, annualized_sharpe: float | None = None
) -> float:
    r = returns.dropna()
    t = len(r)
    if t < 30 or n_trials <= 0:
        return np.nan
    if annualized_sharpe is None:
        annualized_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252)
    expected_max = np.sqrt(-np.log(1 - 0.5 ** (1.0 / n_trials)) / np.log(4))
    if annualized_sharpe <= expected_max:
        return 0.0
    sr_daily = annualized_sharpe / np.sqrt(252)
    skew = r.skew()
    kurt = r.kurtosis() + 3.0
    denom_sq = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)
    if denom_sq <= 0:
        return np.nan
    z = (annualized_sharpe - expected_max) * np.sqrt(t - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def metrics(r: pd.Series):
    cagr, dd = cagr_dd(r)
    vol = r.std() * np.sqrt(252)
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe(r),
        "MaxDD": dd,
        "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
    }


def main():
    ret_a, ret_b, _, _ = build_core_returns(PX, commission_bps=COST_BPS)
    sl_band = build_sleeve_returns(PX, cost_bps=COST_BPS, rates_band=BAND)
    sl_default = build_sleeve_returns(PX, cost_bps=COST_BPS)

    common = (ret_a.index.intersection(ret_b.index)
              .intersection(sl_band.index))
    ret_a, ret_b = ret_a.loc[common], ret_b.loc[common]
    sl_band = sl_band.loc[common]
    sl_default = sl_default.loc[common]

    rates_band = sl_band["rates"]
    rates_default = sl_default["rates"]
    bear = sl_default["bear"]
    cta = sl_default["cta"]

    ens_band = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates_band + 0.2 * bear + 0.2 * cta
    ens_default = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates_default + 0.2 * bear + 0.2 * cta

    train_mask = common <= TRAIN_END
    test_mask = (common >= TEST_START) & (common <= TEST_END)

    print("=" * 100)
    print(f"RATES BAND SINGLE-TRIAL VALIDATION  band={BAND*100:.1f}%  (one pre-registered hypothesis)")
    print("=" * 100)
    print(f"Train: {common[train_mask][0].date()} -> {common[train_mask][-1].date()}")
    print(f"Test:  {common[test_mask][0].date()} -> {common[test_mask][-1].date()}")
    print()

    for label, r, sl in [("default N=2 hysteresis", ens_default, sl_default),
                         ("band 2.5%", ens_band, sl_band)]:
        test_r = r.loc[test_mask].dropna()
        test_rates = sl["rates"].loc[test_mask].dropna()
        m = metrics(test_r)
        m_rates = metrics(test_rates)
        sr = m["Sharpe"]
        psr = probabilistic_sharpe_ratio(test_r)
        dsr = deflated_sharpe_ratio(test_r, n_trials=1, annualized_sharpe=sr)

        print(f"{label}")
        print(f"  Ensemble test Sharpe = {sr:.3f}   MaxDD = {m['MaxDD']:+.1%}   Calmar = {m['Calmar']:.3f}")
        print(f"  Rates  test Sharpe   = {m_rates['Sharpe']:.3f}   MaxDD = {m_rates['MaxDD']:+.1%}   "
              f"turns/yr = {turns_per_year(_rates_weights(PX, band=BAND if 'band' in label else None)):.1f}")
        print(f"  PSR vs 0             = {psr:.1%}")
        print(f"  DSR (n_trials=1)     = {dsr:.1%}")
        print()

    # Full-sample sanity check for stability.
    print("-" * 100)
    print("FULL-SAMPLE STABILITY (for context, not used for selection)")
    print("-" * 100)
    for label, r in [("default", ens_default), ("band 2.5%", ens_band)]:
        m = metrics(r.dropna())
        print(f"{label:18s} Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']:+.1%}  Calmar={m['Calmar']:.3f}")


if __name__ == "__main__":
    main()
