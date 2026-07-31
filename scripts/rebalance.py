"""Daily live runner for the A+B+Diversifier Sleeves strategy on Alpaca.

Designed to be invoked once per trading day after the market close (e.g. via cron
at 16:35 ET). It:

  1. Fetches point-in-time prices for the core and sleeve universes.
  2. Recomputes Strategy A and Strategy B signals.
  3. Builds target sleeve-level weights and decomposes them to individual tickers.
  4. Runs risk guardrails.
  5. Sends orders to Alpaca (paper by default).
  6. Logs weights, orders, and account state.

Environment variables:
  ALPACA_API_KEY      required
  ALPACA_API_SECRET   required
  ALPACA_LIVE=true    required to use live trading (default is paper)

Usage:
  cd Live
  python scripts/rebalance.py [--date YYYY-MM-DD] [--dry-run] [--prefer-yfinance]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Repo root (parent of scripts/) on path so `live.*` imports resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from the repo root.
try:
    from dotenv import load_dotenv
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
    else:
        print(f"WARN: .env file not found at {env_path}")
except Exception as e:
    print(f"WARN: could not load .env: {e}")

# Verify credentials were loaded.
if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
    print("WARN: ALPACA_API_KEY / ALPACA_API_SECRET not set. Check Live/.env")

from live.core_signals import UNIVERSE as CORE_UNIVERSE, build_core_returns
from live.data_feed import fetch_panel, get_last_trading_day
from live.portfolio import (
    SLEEVE_TICKERS,
    SleeveConfig,
    build_live_weights,
    build_sleeve_returns,
    decompose_target_to_tickers,
)
from live.risk import RiskGuard
from live.alpaca_executor import AlpacaExecutor, TargetPortfolio
from live.state import (
    get_peak_equity,
    load_last_weights,
    save_last_weights,
    update_peak_equity,
)


LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_LOG = LOG_DIR / "target_weights.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily live runner for A+B+Diversifier Sleeves")
    p.add_argument("--date", type=str, default=None, help="Run as-of date (YYYY-MM-DD)")
    p.add_argument("--dry-run", action="store_true", help="Do not place live orders")
    p.add_argument("--prefer-yfinance", action="store_true", help="Use yfinance instead of Alpaca")
    return p.parse_args()


def _save_run_log(
    run_date: date,
    target_weights: Dict[str, float],
    orders: list,
    account: Dict[str, float],
    dry_run: bool,
) -> None:
    record = {
        "date": run_date.isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "target_weights": target_weights,
        "orders": [
            {
                "ticker": o.ticker,
                "side": o.side,
                "qty": o.qty,
                "notional": o.notional,
                "status": o.status,
                "message": o.message,
            }
            for o in orders
        ],
        "account": account,
    }
    with WEIGHT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    args = _parse_args()
    run_date = date.fromisoformat(args.date) if args.date else get_last_trading_day()
    end_date = run_date
    start_date = end_date - timedelta(days=365 * 3)

    print(f"[{datetime.now()}] Live runner starting for {run_date} (dry_run={args.dry_run})")

    # 1. Data fetch.
    all_tickers = list(dict.fromkeys(CORE_UNIVERSE + SLEEVE_TICKERS + ["^VIX", "BIL"]))
    prefer_alpaca = not args.prefer_yfinance
    try:
        prices = fetch_panel(all_tickers, start_date, end_date, prefer_alpaca=prefer_alpaca)
    except Exception as e:
        print(f"FATAL: could not fetch price panel: {e}")
        return 1

    # Rename ^VIX to VIX for consistency.
    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})
    if "VIX" not in prices.columns:
        print("FATAL: VIX data missing")
        return 1

    # Ensure cash proxy exists.
    if "BIL" not in prices.columns:
        prices["BIL"] = 100.0

    # 2. Core signals A and B.
    try:
        ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    except Exception as e:
        print(f"FATAL: core signal engine failed: {e}")
        return 1

    # 3. Sleeve returns and target weights.
    sleeve_rets = build_sleeve_returns(prices)
    config = SleeveConfig()
    last_weights = load_last_weights()
    sleeve_weights = build_live_weights(ret_a, ret_b, sleeve_rets, config, last_weights, today=pd.Timestamp(run_date))

    # 4. Decompose to tickers.
    latest_a = weights_a.iloc[-1].fillna(0.0)
    latest_b = weights_b.iloc[-1].fillna(0.0)
    target_tickers = decompose_target_to_tickers(sleeve_weights, latest_a, latest_b, prices)

    print(f"Sleeve weights: {sleeve_weights.to_dict()}")
    print(f"Ticker targets: {target_tickers}")

    # 5. Risk checks.
    peak_equity = get_peak_equity() or 0.0
    guard = RiskGuard(peak_equity=peak_equity)
    if not guard.should_run_today(run_date):
        print("INFO: market closed today; skipping.")
        return 0

    risk = guard.check(prices, target_tickers, live_equity=peak_equity, current_date=run_date)
    if not risk.ok:
        for msg in risk.messages:
            print(f"RISK BLOCK: {msg}")
        return 2

    # 6. Execution.
    try:
        if args.dry_run:
            executor = AlpacaExecutor(client=None)
            account = {
                "equity": 100_000.0,
                "cash": 100_000.0 * (1.0 - sum(abs(v) for v in target_tickers.values())),
                "buying_power": 100_000.0,
                "portfolio_value": 100_000.0,
            }
        else:
            executor = AlpacaExecutor()
            account = executor.get_account()
        equity = account["equity"]

        # Update peak equity with today's account value.
        update_peak_equity(equity)

        target_dollars = {t: w * equity for t, w in target_tickers.items()}
        latest_prices = {t: float(prices[t].iloc[-1]) for t in target_tickers if t in prices.columns}

        target_portfolio = TargetPortfolio(
            date=pd.Timestamp(run_date),
            targets=target_dollars,
            expected_cash=equity * (1.0 - sum(abs(v) for v in target_tickers.values())),
            strategy_weights=sleeve_weights.to_dict(),
            notes=f"A+B+Diversifier live rebalance ({'paper' if executor.paper else 'LIVE'})",
        )
        orders = executor.rebalance(target_portfolio, latest_prices, dry_run=args.dry_run)
    except Exception as e:
        print(f"FATAL: execution failed: {e}")
        return 1

    # 7. Persist state and logging.
    save_last_weights(target_tickers, run_date)
    _save_run_log(run_date, target_tickers, orders, account, args.dry_run)

    print(f"Account equity: ${equity:,.2f}")
    for o in orders:
        print(f"  {o.side:4s} {o.ticker:6s} qty={o.qty:,.4f} notional=${o.notional:,.2f} status={o.status}")
    print(f"[{datetime.now()}] Live runner finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
