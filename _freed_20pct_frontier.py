"""Freed-20% allocation frontier.

Dropping the bear sleeve frees 20% of NAV. This script maps where that 20% can
go along the Sharpe/MaxDD frontier:

  A. 20% BIL ballast       (A20/B20/rates20/cta20/BIL20)
  B. +20% CTA              (A20/B20/rates20/cta40)
  C. +10% CTA / +10% BIL   (A20/B20/rates20/cta30/BIL10)
  D. Baseline for comparison (A20/B20/rates20/bear20/cta20)

All variants use the same A/B/rates sleeve returns. If rates_band is provided,
the rates sleeve uses it; otherwise the default N=2 hysteresis gate is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.core_signals import build_core_returns
from live.portfolio import build_sleeve_returns

PX = pd.read_parquet("cache/_full_panel_2015_2026.parquet")
COST_BPS = 10.0


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
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe(r),
        "MaxDD": dd,
        "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
    }


def year_ret(r: pd.Series, y: int) -> float:
    m = r.index.year == y
    return (1 + r[m]).prod() - 1 if m.sum() else np.nan


def run(rates_band: float | None = None):
    ret_a, ret_b, _, _ = build_core_returns(PX, commission_bps=COST_BPS)
    sleeves = build_sleeve_returns(PX, cost_bps=COST_BPS, rates_band=rates_band)

    common = ret_a.index.intersection(ret_b.index).intersection(sleeves.index)
    ret_a, ret_b = ret_a.loc[common], ret_b.loc[common]
    sleeves = sleeves.loc[common]
    rates = sleeves["rates"]
    bear = sleeves["bear"]
    cta = sleeves["cta"]
    bil = PX.loc[common, "BIL"].pct_change(fill_method=None).fillna(0.0)

    variants = {
        "baseline (bear 20)": (0.2, 0.2, 0.2, 0.2, 0.2, 0.0),
        "BIL ballast (freed 20)": (0.2, 0.2, 0.2, 0.0, 0.2, 0.2),
        "+20% CTA (freed 20)": (0.2, 0.2, 0.2, 0.0, 0.4, 0.0),
        "+10% CTA / +10% BIL (freed 20)": (0.2, 0.2, 0.2, 0.0, 0.3, 0.1),
    }

    rows = []
    for label, (wa, wb, wr, wbear, wcta, wbil) in variants.items():
        r = wa * ret_a + wb * ret_b + wr * rates + wbear * bear + wcta * cta + wbil * bil
        m = metrics(r)
        rows.append({
            "config": label,
            "w_A": wa,
            "w_B": wb,
            "w_rates": wr,
            "w_bear": wbear,
            "w_cta": wcta,
            "w_BIL": wbil,
            **m,
            "2018": year_ret(r, 2018),
            "2020": year_ret(r, 2020),
            "2022": year_ret(r, 2022),
            "2025": year_ret(r, 2025),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 120)
    print("FREED-20% ALLOCATION FRONTIER  --  default rates gate (N=2 hysteresis)")
    print("=" * 120)
    df_default = run(rates_band=None)
    print(df_default.to_string(index=False))

    print()
    print("=" * 120)
    print("FREED-20% ALLOCATION FRONTIER  --  rates band 2.5%")
    print("=" * 120)
    df_band = run(rates_band=0.025)
    print(df_band.to_string(index=False))

    print()
    print("INTERPRETATION:")
    print("  Pick the operating point from live MaxDD tolerance, not the Sharpe argmax.")
    print("  +20% CTA is Sharpe-focused but concentrates CTA exposure (CTA standalone MaxDD ~-38%).")
    print("  BIL ballast is the MaxDD-focused point. +10/10 is the middle.")
    print("  The band rates gate raises all variants; combining it with any freed-20%")
    print("  allocation is a 2-selection combo and needs its own DSR check.")


if __name__ == "__main__":
    main()
