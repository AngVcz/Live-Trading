"""Alpaca execution layer for the A+B+Diversifier Sleeves strategy.

All orders are placed in paper mode unless ``ALPACA_LIVE=true`` is set.
The executor sells first, then buys, to free buying power and minimize margin risk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional Alpaca SDK.
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:  # pragma: no cover
    TradingClient = None  # type: ignore
    OrderSide = None  # type: ignore
    TimeInForce = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent  # live/ -> repo root
LOG_DIR = REPO_ROOT / "logs" / "orders"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_trading_client() -> Optional[TradingClient]:
    if TradingClient is None:
        raise RuntimeError("alpaca-py is not installed")
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
    paper = os.environ.get("ALPACA_LIVE", "").lower() != "true"
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


@dataclass(frozen=True)
class TargetPortfolio:
    """Target dollar allocations and metadata for one rebalance day."""

    date: pd.Timestamp
    targets: Dict[str, float]  # ticker -> dollar target (negative = short)
    expected_cash: float
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class OrderResult:
    ticker: str
    side: str
    qty: float
    notional: float
    status: str
    message: str = ""


class _NoClientSentinel:
    """Sentinel to distinguish 'no client passed' from 'None passed explicitly'."""


class AlpacaExecutor:
    """Place orders to move the account toward a target allocation."""

    def __init__(
        self,
        client: Optional[TradingClient] = _NoClientSentinel,
        fractional: bool = True,
    ):
        self.fractional = fractional
        self.paper = True
        if client is _NoClientSentinel:
            if TradingClient is None:
                raise RuntimeError("alpaca-py is not installed")
            self.client = _get_trading_client()
            self.paper = getattr(self.client, "_paper", True)
        else:
            self.client = client
            if client is not None:
                self.paper = getattr(client, "_paper", True)

    def get_account(self) -> Dict[str, float]:
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_positions(self) -> Dict[str, float]:
        """Return current market value per ticker."""
        if self.client is None:
            return {}
        positions = self.client.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    def _qty_for_notional(self, notional: float, price: float) -> float:
        if price <= 0:
            return 0.0
        if self.fractional:
            return notional / price
        shares = int(notional / price)
        return float(shares)

    def rebalance(
        self,
        target: TargetPortfolio,
        prices: Dict[str, float],
        dry_run: bool = True,
    ) -> List[OrderResult]:
        """
        Generate and optionally place orders to reach ``target``.

        Parameters
        ----------
        target : TargetPortfolio
        prices : dict[str, float]
            Latest close prices for every ticker in the target.
        dry_run : bool
            If True, return intended orders without sending them to Alpaca.

        Returns
        -------
        list[OrderResult]
        """
        if self.client is None:
            # Dry-run path without a live client.
            account = {
                "equity": sum(abs(v) for v in target.targets.values()) + target.expected_cash,
                "cash": target.expected_cash,
                "buying_power": sum(abs(v) for v in target.targets.values()) + target.expected_cash,
                "portfolio_value": sum(abs(v) for v in target.targets.values()) + target.expected_cash,
            }
            positions = {}
        else:
            account = self.get_account()
            positions = self.get_positions()
        equity = account["equity"]

        results: List[OrderResult] = []
        orders: List[Dict[str, Any]] = []

        # Normalize tickers to upper case.
        current = {k.upper(): v for k, v in positions.items()}
        target_clean = {k.upper(): v for k, v in target.targets.items()}
        all_tickers = sorted(set(current.keys()) | set(target_clean.keys()))

        for t in all_tickers:
            price = prices.get(t, 0.0)
            if price <= 0:
                results.append(OrderResult(t, "skip", 0.0, 0.0, "error", f"no price for {t}"))
                continue
            target_dollar = target_clean.get(t, 0.0)
            current_dollar = current.get(t, 0.0)
            delta = target_dollar - current_dollar
            if abs(delta) < 0.01:
                continue
            side = "BUY" if delta > 0 else "SELL"
            qty = self._qty_for_notional(abs(delta), price)
            if qty <= 0:
                continue
            orders.append({
                "ticker": t,
                "side": side,
                "qty": qty,
                "notional": abs(delta),
                "price": price,
            })

        # Sell first, then buy.
        sell_orders = [o for o in orders if o["side"] == "SELL"]
        buy_orders = [o for o in orders if o["side"] == "BUY"]
        ordered = sell_orders + buy_orders

        for o in ordered:
            side_label = o["side"]
            if dry_run:
                results.append(OrderResult(
                    o["ticker"], side_label, o["qty"], o["notional"], "dry_run"
                ))
                continue
            try:
                alpaca_side = OrderSide.BUY if o["side"] == "BUY" else OrderSide.SELL
                # ponytail: fractional orders must be DAY on Alpaca (42210000);
                # whole-share keeps GTC. Daily rebalance wants DAY anyway.
                qty = Decimal(str(o["qty"])).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
                tif = TimeInForce.DAY if (self.fractional and qty % 1 != 0) else TimeInForce.GTC
                req = MarketOrderRequest(
                    symbol=o["ticker"],
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=tif,
                )
                submitted = self.client.submit_order(req)
                results.append(OrderResult(
                    o["ticker"],
                    side_label,
                    o["qty"],
                    o["notional"],
                    str(submitted.status),
                ))
            except Exception as e:
                results.append(OrderResult(
                    o["ticker"], side_label, o["qty"], o["notional"], "error", str(e)
                ))

        self._log_rebalance(target, account, positions, results)
        return results

    def _log_rebalance(
        self,
        target: TargetPortfolio,
        account: Dict[str, float],
        positions: Dict[str, float],
        results: List[OrderResult],
    ) -> None:
        path = LOG_DIR / f"orders_{target.date.strftime('%Y%m%d')}.csv"
        rows = []
        for r in results:
            rows.append({
                "date": target.date,
                "ticker": r.ticker,
                "side": r.side,
                "qty": r.qty,
                "notional": r.notional,
                "status": r.status,
                "message": r.message,
                "account_equity": account["equity"],
                "account_cash": account["cash"],
                "paper": self.paper,
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
