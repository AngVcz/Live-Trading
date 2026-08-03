"""Alpaca execution layer for the A+B+Diversifier Sleeves strategy.

All orders are placed in paper mode unless ``ALPACA_LIVE=true`` is set.
The executor sells first, then buys, to free buying power and minimize margin risk.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional Alpaca SDK.
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:  # pragma: no cover
    TradingClient = None  # type: ignore
    OrderSide = None  # type: ignore
    TimeInForce = None  # type: ignore
    LimitOrderRequest = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent  # live/ -> repo root
LOG_DIR = REPO_ROOT / "logs" / "orders"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Wash-sale awareness (WARN only, no tax-lot logic). Tracked inverse/vol ETFs:
# every SELL of one of these is conservatively recorded as a realized-loss date;
# a BUY of the same symbol within 30 calendar days emits a warning (never blocks).
WASH_SALE_SYMBOLS = frozenset({"SH", "PSQ", "VIXY"})
WASH_SALE_WINDOW_DAYS = 30
WASH_SALE_PATH = REPO_ROOT / "logs" / "wash_sale.json"


def load_wash_sale() -> Dict[str, List[str]]:
    if not WASH_SALE_PATH.exists():
        return {}
    try:
        return json.loads(WASH_SALE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_wash_sale(state: Dict[str, List[str]]) -> None:
    WASH_SALE_PATH.parent.mkdir(parents=True, exist_ok=True)
    WASH_SALE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_realized_loss(symbol: str, fill_date: Any) -> None:
    """Record a realized-loss date for a wash-sale-tracked symbol.

    Conservative and mechanical: treats every SELL of SH/PSQ/VIXY as a loss
    event (no tax-lot/basis logic). This intentionally OVER-WARNS on re-entries
    after a profitable sell, but is WARN-only and carries no tax semantics.
    Dates older than the 30-day window relative to ``fill_date`` are pruned
    to keep the JSON bounded.
    """
    sym = symbol.upper()
    if sym not in WASH_SALE_SYMBOLS:
        return
    as_of = pd.Timestamp(fill_date).date()
    cutoff = as_of - pd.Timedelta(days=WASH_SALE_WINDOW_DAYS)
    state = load_wash_sale()
    dates = [d for d in state.get(sym, []) if pd.Timestamp(d).date() >= cutoff]
    iso = as_of.isoformat()
    if iso not in dates:
        dates.append(iso)
    state[sym] = dates
    save_wash_sale(state)


def _wash_sale_loss_within(symbol: str, as_of: Any) -> Optional[str]:
    """ISO date of the most recent recorded realized loss within 30 cal days, else None."""
    sym = symbol.upper()
    if sym not in WASH_SALE_SYMBOLS:
        return None
    state = load_wash_sale()
    dates = state.get(sym, [])
    if not dates:
        return None
    as_of_d = pd.Timestamp(as_of).date()
    for iso in sorted(dates, reverse=True):
        d = pd.Timestamp(iso).date()
        delta_days = (as_of_d - d).days
        if 0 <= delta_days <= WASH_SALE_WINDOW_DAYS:
            return iso
    return None


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
        illiquid_symbols: Optional[Any] = None,
        illiquid_buffer: float = 0.005,
    ):
        self.fractional = fractional
        # Optional limit-price set for illiquid names that gap on market open
        # (e.g. KMLM, VIXY, PSQ). Default empty -> plain market orders for all.
        self.illiquid_symbols = set(illiquid_symbols) if illiquid_symbols else set()
        self.illiquid_buffer = float(illiquid_buffer)
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
            held_qty: Dict[str, float] = {}
        else:
            account = self.get_account()
            raw_positions = self.client.get_all_positions()
            positions = {p.symbol: float(p.market_value) for p in raw_positions}
            # ponytail: held share qty per symbol, to floor SELL orders (below).
            held_qty = {p.symbol.upper(): float(p.qty)
                        for p in raw_positions if getattr(p, "qty", None) is not None}
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
            # Delta deadband: $0.01 per-share rounding/noise threshold — skip
            # ~zero deltas (the rounding noise floor, NOT a skip-size gate).
            if abs(delta) < 0.01:
                continue
            notional = abs(delta)
            # Min-notional skip: $1 — too small to bother sending. This is a
            # coarser skip-decision gate ON TOP of the $0.01 delta deadband:
            # an order can clear the $0.01 deadband but still be sub-$1 noise.
            if notional < 1.0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            qty = self._qty_for_notional(notional, price)
            if qty <= 0:
                continue
            # ponytail: never SELL more shares than held. The runner often
            # executes intraday, where latest_prices is the last CLOSE but
            # current_dollar is the live mark-to-market; notional/price then
            # exceeds the held qty and Alpaca rejects the SELL as insufficient.
            # Floor to the held qty so a full sell clears the position exactly.
            if side == "SELL":
                qty = min(qty, held_qty.get(t, qty))
            orders.append({
                "ticker": t,
                "side": side,
                "qty": qty,
                "notional": notional,
                "price": price,
            })

        # Sell first, then buy.
        # TODO(buying-power): do NOT re-fetch buying power between sell and buy
        # batches here. Sells land asynchronously and may not have settled; a
        # half-correct re-fetch would race the fill stream and could over/under-
        # state available buying power. Leave the single pre-rebalance fetch.
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
            # Wash-sale awareness (WARN only, never blocks): a BUY of an inverse
            # /vol ETF within 30 days of a recorded realized loss on the same
            # symbol. Emitted before submit so the order still goes through.
            if side_label == "BUY":
                loss_date = _wash_sale_loss_within(o["ticker"], target.date)
                if loss_date is not None:
                    wmsg = (f"wash-sale: BUY {o['ticker']} within 30d of "
                            f"realized loss on {loss_date}")
                    warnings.warn(wmsg)
                    print(f"[WARN] {wmsg}")
            submit_ok = False
            try:
                alpaca_side = OrderSide.BUY if o["side"] == "BUY" else OrderSide.SELL
                qty = Decimal(str(o["qty"])).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
                # Daily rebalance: no order should sit open across days, so
                # whole-share AND fractional market orders use DAY. (Fractional
                # already required DAY on Alpaca; whole-share now matches.)
                tif = TimeInForce.DAY
                submitted = self._submit_order(o, alpaca_side, qty, tif)
                status_str = str(submitted.status)
                results.append(OrderResult(
                    o["ticker"],
                    side_label,
                    o["qty"],
                    o["notional"],
                    status_str,
                ))
                submit_ok = True
                # Non-fill observation (WARN, not raise): a DAY limit may be
                # accepted but sit unfilled and expire at EOD; a market order
                # that doesn't immediately fill is surfaced too. The safety net
                # is next-day drift recomputing and resubmitting the delta.
                if status_str != "filled":
                    nf = (f"{o['ticker']} order not filled (status={status_str}); "
                          f"will be re-corrected by next-day drift")
                    warnings.warn(nf)
                    print(f"[WARN] {nf}")
            except Exception as e:
                results.append(OrderResult(
                    o["ticker"], side_label, o["qty"], o["notional"], "error", str(e)
                ))
            # Record realized-loss date on SELL of wash-sale-tracked symbols.
            # Isolated so a wash_sale.json write failure never double-logs or
            # crashes the order path; only on a successful submit.
            if submit_ok and side_label == "SELL" and o["ticker"] in WASH_SALE_SYMBOLS:
                try:
                    record_realized_loss(o["ticker"], target.date)
                except Exception as we:
                    warnings.warn(f"wash-sale record failed for {o['ticker']}: {we}")

        self._log_rebalance(target, account, positions, results)
        return results

    def _submit_order(self, o, alpaca_side, qty, tif):
        """Submit one order, using a limit price for the illiquid set.

        For symbols in ``self.illiquid_symbols`` a limit order at
        ``close*(1 ∓ buffer)`` is attempted first (sell above close, buy below).
        If the SDK rejects fractional+limit+DAY (or anything else about the
        limit request), fall back to a plain market order for that symbol.
        """
        sym = o["ticker"]
        price = o["price"]
        if sym in self.illiquid_symbols and LimitOrderRequest is not None:
            try:
                if o["side"] == "BUY":
                    lp = price * (1.0 - self.illiquid_buffer)
                else:
                    lp = price * (1.0 + self.illiquid_buffer)
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=tif,
                    limit_price=lp,
                )
                # TODO(limit-never-fill): a DAY limit may expire unfilled at EOD;
                # no in-day cancel/fallback-to-market is performed here. Next-day
                # drift recomputes the delta and resubmits, which is the safety
                # net. The caller also warns on any non-"filled" submit status.
                return self.client.submit_order(req)
            except Exception:
                # ponytail: fractional+limit+DAY is rejected by Alpaca for
                # some combos; fall back to a plain market order.
                req = MarketOrderRequest(
                    symbol=sym, qty=qty, side=alpaca_side, time_in_force=tif,
                )
                return self.client.submit_order(req)
        req = MarketOrderRequest(
            symbol=sym, qty=qty, side=alpaca_side, time_in_force=tif,
        )
        return self.client.submit_order(req)

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
        # APPEND on same-day re-runs (preserve earlier runs, don't overwrite).
        write_header = not path.exists()
        try:
            df.to_csv(path, index=False, mode="a", header=write_header)
        except Exception as le:
            # ponytail: logging must never crash the live runner after orders
            # were already submitted (e.g. a Windows file lock by another proc).
            warnings.warn(f"rebalance log write failed for {path}: {le}")
            print(f"[WARN] rebalance log write failed for {path}: {le}")
