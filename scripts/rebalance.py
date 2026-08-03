"""Daily live runner for the A+B+Diversifier Sleeves strategy on Alpaca.

Designed to be invoked once per trading day after the market close (e.g. via cron
at 16:35 ET). It:

  1. Fetches point-in-time prices for the core and sleeve universes.
  2. Recomputes Strategy A and Strategy B signals.
  3. Builds target sleeve-level weights and decomposes them to individual tickers.
  4. Runs risk guardrails.
  5. Sends orders to Alpaca (paper by default).
  6. Logs weights, orders, and account state.

With ``--profile {aggressive,balanced,passive}`` it instead trades the discretionary
profile chosen from the 07:30 morning report
(``logs/discretionary_<date>.json``); weights come from the report, not from the
systematic engine.

Environment variables:
  ALPACA_API_KEY      required
  ALPACA_API_SECRET   required
  ALPACA_LIVE=true    required to use live trading (default is paper)

Usage:
  cd Live
  python scripts/rebalance.py [--date YYYY-MM-DD] [--dry-run] [--prefer-yfinance]
  python scripts/rebalance.py --profile balanced [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from live.discretionary import PROFILES, apply_profile
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

# ponytail: default 5% ticker-level drift threshold; trades are skipped when the
# current book is within this band of the targets (unless it's the annual window).
DEFAULT_DRIFT_THRESHOLD = 0.05


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily live runner for A+B+Diversifier Sleeves")
    p.add_argument("--date", type=str, default=None, help="Run as-of date (YYYY-MM-DD)")
    p.add_argument("--dry-run", action="store_true", help="Do not place live orders")
    p.add_argument("--prefer-yfinance", action="store_true", help="Use yfinance instead of Alpaca")
    p.add_argument("--profile", choices=list(PROFILES), default=None,
                   help="Trade a discretionary profile from logs/discretionary_<date>.json")
    p.add_argument("--weights", type=str, default=None,
                   help="Custom ticker->weight JSON override (e.g. '{\"BIL\":0.5867,...}'); "
                        "bypasses both the systematic engine and --profile. Risk guardrails still apply.")
    p.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD,
                   help="Ticker-level drift threshold above which a rebalance is forced (default 0.05)")
    return p.parse_args()


def compute_systematic_targets(run_date: date, prefer_alpaca: bool = True) -> Dict:
    """Run the data-fetch → core-signals → sleeve-weights → decompose pipeline.

    Returns a dict with keys: sleeve_weights (pd.Series), ticker_weights (dict),
    prices (pd.DataFrame), weights_a_last (pd.Series), weights_b_last (pd.Series).
    Shared by the systematic path and by morning_report.py.
    """
    end_date = run_date
    start_date = end_date - timedelta(days=365 * 3)

    all_tickers = list(dict.fromkeys(CORE_UNIVERSE + SLEEVE_TICKERS + ["^VIX", "BIL"]))
    prices = fetch_panel(all_tickers, start_date, end_date, prefer_alpaca=prefer_alpaca)

    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})
    if "VIX" not in prices.columns:
        raise RuntimeError("VIX data missing")
    if "BIL" not in prices.columns:
        prices["BIL"] = 100.0

    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve_rets = build_sleeve_returns(prices)
    config = SleeveConfig()
    last_weights = load_last_weights()
    sleeve_weights = build_live_weights(
        ret_a, ret_b, sleeve_rets, config, last_weights, today=pd.Timestamp(run_date)
    )

    latest_a = weights_a.iloc[-1].fillna(0.0)
    latest_b = weights_b.iloc[-1].fillna(0.0)
    ticker_weights = decompose_target_to_tickers(sleeve_weights, latest_a, latest_b, prices)

    return {
        "sleeve_weights": sleeve_weights,
        "ticker_weights": ticker_weights,
        "prices": prices,
        "weights_a_last": latest_a,
        "weights_b_last": latest_b,
    }


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


def _load_profile_tickers(run_date: date, profile: str) -> Dict[str, float]:
    """Read the chosen profile's ticker weights from the morning report JSON."""
    path = LOG_DIR / f"discretionary_{run_date.isoformat()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"discretionary report not found: {path}. Run scripts/morning_report.py "
            f"--date {run_date.isoformat()} first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["profiles"][profile]["tickers"]


# --- Drift gate (ticker-level) ----------------------------------------------
def is_annual_rebalance_window(run_date: date) -> bool:
    """Annual scheduled rebalance window (matches build_live_weights: first 5 days of Jan)."""
    return run_date.month == 1 and run_date.day <= 5


def positions_to_weights(positions_value: Dict[str, float], equity: float) -> Dict[str, float]:
    """Convert market values per ticker to weights using the same equity base."""
    if equity <= 0:
        return {t: 0.0 for t in positions_value}
    return {t: v / equity for t, v in positions_value.items()}


def compute_max_drift(
    target_tickers: Dict[str, float], current_weights: Dict[str, float]
) -> float:
    """Max absolute weight deviation between targets and the current book (weight-based)."""
    tickers = set(target_tickers) | set(current_weights)
    if not tickers:
        return 0.0
    return max(abs(target_tickers.get(t, 0.0) - current_weights.get(t, 0.0)) for t in tickers)


def drift_gate_skips(
    target_tickers: Dict[str, float],
    current_positions_value: Dict[str, float],
    equity: float,
    drift_threshold: float,
    run_date: date,
) -> bool:
    """True when execution should be skipped: drift within threshold AND not in the annual window."""
    current_weights = positions_to_weights(current_positions_value, equity)
    max_drift = compute_max_drift(target_tickers, current_weights)
    return max_drift <= drift_threshold and not is_annual_rebalance_window(run_date)


def decide_and_execute(
    executor: AlpacaExecutor,
    target_tickers: Dict[str, float],
    sleeve_weights,
    prices: pd.DataFrame,
    run_date: date,
    account: Dict[str, float],
    drift_threshold: float,
    dry_run: bool,
) -> Tuple[List, bool, bool]:
    """Apply the ticker-level drift gate; execute via ``executor.rebalance`` unless skipping.

    Always logs target weights + orders to WEIGHT_LOG (audit trail, tagged dry_run).
    Persists ``last_weights`` ONLY on a clean, non-dry-run, error-free run. When the
    current book is within ``drift_threshold`` of the targets (and it is not the annual
    rebalance window) NO orders are placed; the targets are still logged.

    Returns ``(orders, skipped, had_error)``. ``had_error`` is True if any target or
    held ticker could not be priced (a rejected/missing order would otherwise silently
    no-op) or any submitted order came back with ``status == "error"``. The caller must
    treat ``had_error`` as fatal: skip peak-equity persistence and exit non-zero.
    """
    equity = account["equity"]
    current_positions = executor.get_positions()
    skipped = drift_gate_skips(target_tickers, current_positions, equity, drift_threshold, run_date)

    sw_dict = sleeve_weights.to_dict() if hasattr(sleeve_weights, "to_dict") else dict(sleeve_weights)

    orders: List = []
    had_error = False

    if skipped:
        current_weights = positions_to_weights(current_positions, equity)
        max_drift = compute_max_drift(target_tickers, current_weights)
        print(f"DRIFT: max drift {max_drift:.2%} <= {drift_threshold:.2%}; "
              f"skipping execution (annual_window={is_annual_rebalance_window(run_date)})")
        print(f"Ticker targets (logged, not traded): {target_tickers}")
    else:
        target_dollars = {t: w * equity for t, w in target_tickers.items()}
        # ponytail: price the UNION of target + current book so off-target holdings
        # (a --weights/--profile target that is a subset of the held book) can be sold.
        price_tickers = set(target_tickers) | set(current_positions)
        latest_prices = {t: float(prices[t].iloc[-1]) for t in price_tickers if t in prices.columns}
        # ponytail #3/#9: a missing TARGET price -> a rejected order; a missing HELD
        # price -> a liquidation that silently no-ops and leaves the position forever.
        # Both must be fatal, not silent skips. The risk guard already validated target
        # tickers; this additionally covers held tickers outside the fetched universe.
        unpriced = sorted(
            t for t in price_tickers
            if t not in latest_prices
            or not math.isfinite(latest_prices[t])
            or latest_prices[t] <= 0
        )
        if unpriced:
            print(f"FATAL: no finite price for {len(unpriced)} ticker(s): {unpriced}")
            had_error = True
        else:
            target_portfolio = TargetPortfolio(
                date=pd.Timestamp(run_date),
                targets=target_dollars,
                expected_cash=equity * (1.0 - sum(abs(v) for v in target_tickers.values())),
                strategy_weights=sw_dict,
                notes=f"A+B+Diversifier live rebalance ({'paper' if executor.paper else 'LIVE'})",
            )
            orders = executor.rebalance(target_portfolio, latest_prices, dry_run=dry_run)
            # ponytail #1: any order that errored makes the run fatal -- never save
            # state as if the target was reached when orders were rejected.
            errored = [o for o in orders if getattr(o, "status", "") == "error"]
            if errored:
                had_error = True
                print(f"FATAL: {len(errored)} order(s) errored: "
                      f"{[(o.ticker, o.message) for o in errored]}")

    # Run-log always (audit, tagged dry_run). last_weights only on a clean, real run:
    # order errors (#1) and dry-runs (#7) must not mutate production state.
    _save_run_log(run_date, target_tickers, orders, account, dry_run)
    if not dry_run and not had_error:
        save_last_weights(target_tickers, run_date)
    return orders, skipped, had_error


def main() -> int:
    args = _parse_args()
    run_date = date.fromisoformat(args.date) if args.date else get_last_trading_day()
    prefer_alpaca = not args.prefer_yfinance

    print(f"[{datetime.now()}] Live runner starting for {run_date} "
          f"(dry_run={args.dry_run}, profile={args.profile})")

    # Fail fast on weekends/holidays BEFORE the expensive data-fetch/signal pipeline.
    if not RiskGuard().should_run_today(run_date):
        print("INFO: market closed today; skipping.")
        return 0

    # 1-4. Systematic targets (also gives us prices for the profile path).
    try:
        sys_targets = compute_systematic_targets(run_date, prefer_alpaca=prefer_alpaca)
    except Exception as e:
        print(f"FATAL: systematic pipeline failed: {e}")
        return 1
    prices = sys_targets["prices"]

    if args.weights:
        try:
            target_tickers = {k: float(v) for k, v in json.loads(args.weights).items()}
        except Exception as e:
            print(f"FATAL: could not parse --weights JSON: {e}")
            return 1
        # ponytail: no sleeve vector for a custom override -> empty Series. The
        # RiskGuard A+B 70% check then falls back to the ticker-keyed lookup (0),
        # so a discretionary override is responsible for its own sleeve budget.
        sleeve_weights = pd.Series(dtype=float)
        print(f"Using custom --weights override ({len(target_tickers)} tickers).")
    elif args.profile:
        try:
            target_tickers = _load_profile_tickers(run_date, args.profile)
        except Exception as e:
            print(f"FATAL: could not load profile '{args.profile}': {e}")
            return 1
        sleeve_weights = apply_profile(args.profile)
        print(f"Using discretionary profile '{args.profile}' "
              f"({len(target_tickers)} tickers).")
    else:
        target_tickers = sys_targets["ticker_weights"]
        sleeve_weights = sys_targets["sleeve_weights"]

    print(f"Sleeve weights: {sleeve_weights.to_dict()}")
    print(f"Ticker targets: {target_tickers}")

    # 5. Risk checks -- fetch REAL equity BEFORE guard.check so the drawdown breaker
    # sees the actual current equity, not the persisted peak.
    if args.dry_run:
        # Dry-run: do not touch Alpaca; use a 100k fallback equity so the breaker still
        # reports what it WOULD do. Never persist the peak from this synthetic equity.
        executor = AlpacaExecutor(client=None)
        account = {
            "equity": 100_000.0,
            "cash": 100_000.0 * (1.0 - sum(abs(v) for v in target_tickers.values())),
            "buying_power": 100_000.0,
            "portfolio_value": 100_000.0,
        }
        live_equity = account["equity"]
    else:
        try:
            executor = AlpacaExecutor()
            account = executor.get_account()
            live_equity = account["equity"]
        except Exception as e:
            print(f"FATAL: could not fetch account equity ({e})")
            return 1

    # Seed the peak: first-ever run seeds to today's equity rather than 0. In dry-run
    # we only read the persisted peak for seeding; we never write it.
    peak_equity = get_peak_equity() or live_equity
    guard = RiskGuard(peak_equity=peak_equity)
    risk = guard.check(
        prices, target_tickers,
        live_equity=live_equity,
        current_date=run_date,
        sleeve_weights=sleeve_weights,
    )
    if not risk.ok:
        for msg in risk.messages:
            print(f"RISK BLOCK: {msg}")
        return 2

    # 6. Drift gate + execution.
    try:
        orders, skipped, had_error = decide_and_execute(
            executor=executor,
            target_tickers=target_tickers,
            sleeve_weights=sleeve_weights,
            prices=prices,
            run_date=run_date,
            account=account,
            drift_threshold=args.drift_threshold,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"FATAL: execution failed: {e}")
        return 1

    # 7. Persist peak equity ONLY on a clean, non-dry-run, error-free run (never from
    # the hard-coded 100k dry-run equity, and never after an order error -- the book
    # did not reach the target, so don't record a peak as if it did).
    if not args.dry_run and not had_error:
        update_peak_equity(live_equity)

    print(f"Account equity: ${live_equity:,.2f}")
    if skipped:
        print(f"[{datetime.now()}] Live runner finished (drift within band; no orders placed)")
    else:
        for o in orders:
            print(f"  {o.side:4s} {o.ticker:6s} qty={o.qty:,.4f} "
                  f"notional=${o.notional:,.2f} status={o.status}")
    if had_error:
        print(f"[{datetime.now()}] Live runner finished with ORDER ERRORS; "
              f"state NOT saved, peak NOT updated")
        return 1
    if not skipped:
        print(f"[{datetime.now()}] Live runner finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())