"""True walkforward validation of the SMA200 band rates gate.

The band-2.5% result in batch-1 and batch-2 was selected on the 2020-2025 window,
so n_trials=1 overstates confidence. This script uses rolling 5-year train / 1-year
test windows: the band width is chosen inside each train window (on data the test
year never saw) and evaluated forward. This is the honest validation that either
promotes band 2.5% to a shippable second change or kills it.

For each train window we search a small grid of band widths and pick the one that
maximizes the Sharpe/MaxDD frontier (or, for comparison, the Sharpe argmax). The
chosen band is then applied to the following year. We also run the default N=2
hysteresis gate on the same windows as a benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from live.core_signals import build_core_returns
from live.portfolio import build_sleeve_returns

PX = pd.read_parquet("cache/_full_panel_2015_2026.parquet")
COST_BPS = 10.0
BAND_GRID = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
TRAIN_YEARS = 5
TEST_YEARS = 1


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan


def metrics(r: pd.Series):
    r = r.dropna()
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1 if yrs else np.nan
    vol = r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe(r), "MaxDD": dd,
            "Calmar": cagr / abs(dd) if dd < 0 else np.nan}


def frontier_pick(df: pd.DataFrame) -> pd.Series:
    """Pick the non-dominated point with highest Sharpe."""
    dominated = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (df.iloc[j]["Sharpe"] >= df.iloc[i]["Sharpe"]
                    and df.iloc[j]["MaxDD"] >= df.iloc[i]["MaxDD"]
                    and (df.iloc[j]["Sharpe"] > df.iloc[i]["Sharpe"]
                         or df.iloc[j]["MaxDD"] > df.iloc[i]["MaxDD"])):
                dominated[i] = True
                break
    return df.loc[~dominated].sort_values("Sharpe", ascending=False).iloc[0]


def make_windows(index: pd.DatetimeIndex):
    start = index.min()
    end = index.max()
    windows = []
    first_test_year = start.year + TRAIN_YEARS
    for test_year in range(first_test_year, end.year + 1):
        train_start = start
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = min(pd.Timestamp(f"{test_year}-12-31"), end)
        if test_start >= end:
            break
        if index[(index >= train_start) & (index <= train_end)].empty:
            continue
        if index[(index >= test_start) & (index <= test_end)].empty:
            continue
        windows.append((train_start, train_end, test_start, test_end))
    return windows


def main():
    ret_a, ret_b, _, _ = build_core_returns(PX, commission_bps=COST_BPS)
    sl_default = build_sleeve_returns(PX, cost_bps=COST_BPS)

    common = ret_a.index.intersection(ret_b.index).intersection(sl_default.index)
    ret_a, ret_b = ret_a.loc[common], ret_b.loc[common]
    sl_default = sl_default.loc[common]
    bear = sl_default["bear"]
    cta = sl_default["cta"]
    bil = PX.loc[common, "BIL"].pct_change(fill_method=None).fillna(0.0)

    windows = make_windows(common)
    print("=" * 120)
    print("RATES BAND TRUE WALKFORWARD  --  5-year train / 1-year test")
    print("=" * 120)
    print(f"Band grid: {BAND_GRID}")
    print(f"Windows: {len(windows)}")
    print()

    rows = []
    for train_start, train_end, test_start, test_end in windows:
        train_mask = (common >= train_start) & (common <= train_end)
        test_mask = (common >= test_start) & (common <= test_end)

        # In each train window, evaluate every band on the ensemble with bear=20, cta=20.
        best_band = None
        best_sharpe = -np.inf
        best_frontier = None

        train_results = []
        for band in BAND_GRID:
            sl = build_sleeve_returns(PX, cost_bps=COST_BPS, rates_band=band)
            rates = sl["rates"].loc[common]
            ens = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates + 0.2 * bear + 0.2 * cta
            m = metrics(ens.loc[train_mask].dropna())
            train_results.append({"band": band, **m})

        train_df = pd.DataFrame(train_results)
        pick = frontier_pick(train_df)
        chosen_band = float(pick["band"])

        # Evaluate chosen band on the test year.
        sl_chosen = build_sleeve_returns(PX, cost_bps=COST_BPS, rates_band=chosen_band)
        rates_chosen = sl_chosen["rates"].loc[common]
        ens_chosen = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates_chosen + 0.2 * bear + 0.2 * cta
        m_chosen = metrics(ens_chosen.loc[test_mask].dropna())

        # Default hysteresis on the same test year.
        rates_default = sl_default["rates"]
        ens_default = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates_default + 0.2 * bear + 0.2 * cta
        m_default = metrics(ens_default.loc[test_mask].dropna())

        rows.append({
            "train": f"{train_start.date()}-{train_end.date()}",
            "test": f"{test_start.date()}-{test_end.date()}",
            "chosen_band": chosen_band,
            "oos_band_Sharpe": m_chosen["Sharpe"],
            "oos_band_MaxDD": m_chosen["MaxDD"],
            "oos_default_Sharpe": m_default["Sharpe"],
            "oos_default_MaxDD": m_default["MaxDD"],
            "delta_Sharpe": m_chosen["Sharpe"] - m_default["Sharpe"],
            "train_Sharpe": pick["Sharpe"],
            "train_MaxDD": pick["MaxDD"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Aggregate walkforward performance (concatenate all test years, no overlap).
    band_oos_rets = []
    default_oos_rets = []
    for _, row in df.iterrows():
        parts = row["test"].split("-")
        # test string is "YYYY-MM-DD-YYYY-MM-DD"; split on the date boundary.
        test_start = pd.Timestamp(f"{parts[0]}-{parts[1]}-{parts[2]}")
        test_end = pd.Timestamp(f"{parts[3]}-{parts[4]}-{parts[5]}")
        test_mask = (common >= test_start) & (common <= test_end)
        band = float(row["chosen_band"])
        sl = build_sleeve_returns(PX, cost_bps=COST_BPS, rates_band=band)
        rates = sl["rates"].loc[common]
        ens_band = 0.2 * ret_a + 0.2 * ret_b + 0.2 * rates + 0.2 * bear + 0.2 * cta
        ens_default = 0.2 * ret_a + 0.2 * ret_b + 0.2 * sl_default["rates"] + 0.2 * bear + 0.2 * cta
        band_oos_rets.append(ens_band.loc[test_mask].dropna())
        default_oos_rets.append(ens_default.loc[test_mask].dropna())

    band_full = pd.concat(band_oos_rets).sort_index()
    default_full = pd.concat(default_oos_rets).sort_index()
    # Drop any accidental duplicates at window boundaries.
    band_full = band_full[~band_full.index.duplicated(keep="first")]
    default_full = default_full[~default_full.index.duplicated(keep="first")]

    print()
    print("=" * 120)
    print("AGGREGATE WALKFORWARD OOS (all test years concatenated)")
    print("=" * 120)
    mb = metrics(band_full)
    md = metrics(default_full)
    print(f"Band-chosen   Sharpe={mb['Sharpe']:.3f}  MaxDD={mb['MaxDD']:+.1%}  Calmar={mb['Calmar']:.3f}")
    print(f"Default N=2   Sharpe={md['Sharpe']:.3f}  MaxDD={md['MaxDD']:+.1%}  Calmar={md['Calmar']:.3f}")
    print(f"Delta         Sharpe={mb['Sharpe'] - md['Sharpe']:+.3f}  MaxDD={(mb['MaxDD'] - md['MaxDD']):+.1%}")

    print()
    print("STABILITY: chosen bands per window")
    print(df["chosen_band"].value_counts().sort_index().to_string())

    print()
    print("INTERPRETATION:")
    print("  • If band-chosen aggregate OOS beats default, the band gate is real.")
    print("  • If not, keep rates_band off by default; the bear drop is the shippable win.")
    print("  • DSR/PSR for the band gate should be reported with n_trials = len(BAND_GRID)")
    print(f"    = {len(BAND_GRID)} per window, because the width is selected inside each train window.")


if __name__ == "__main__":
    main()
