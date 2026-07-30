"""Risk guardrails for the live strategy runner."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class RiskCheckResult:
    ok: bool
    messages: List[str]


class RiskGuard:
    """Point-in-time safety checks before executing any trade."""

    def __init__(
        self,
        max_drawdown_pct: float = -10.0,
        max_single_position_pct: float = 0.50,
        max_sleeve_pct: float = 0.70,
        stale_data_days: int = 2,
        log_dir: Optional[Path] = None,
        peak_equity: Optional[float] = None,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_single_position_pct = max_single_position_pct
        self.max_sleeve_pct = max_sleeve_pct
        self.stale_data_days = stale_data_days
        self.log_dir = log_dir
        self.peak_equity = peak_equity

    def check(
        self,
        prices: pd.DataFrame,
        target_tickers: Dict[str, float],
        live_equity: Optional[float] = None,
        current_date: Optional[date] = None,
    ) -> RiskCheckResult:
        messages: List[str] = []
        today = current_date or date.today()
        last_price_date = prices.index[-1].date()

        # 1. Stale data check.
        if (today - last_price_date).days > self.stale_data_days:
            messages.append(
                f"FAIL: price data stale (last {last_price_date}, today {today})"
            )

        # 2. Ticker availability.
        for t, w in target_tickers.items():
            if abs(w) > 1e-6 and t not in prices.columns:
                messages.append(f"FAIL: target ticker {t} not in price panel")

        # 3. Single position limit.
        for t, w in target_tickers.items():
            if w > self.max_single_position_pct:
                messages.append(
                    f"FAIL: position {t} weight {w:.2%} exceeds max {self.max_single_position_pct:.2%}"
                )

        # 4. Sleeve concentration (A+B combined should not dominate).
        sleeve_core = target_tickers.get("A", 0.0) + target_tickers.get("B", 0.0)
        if sleeve_core > self.max_sleeve_pct:
            messages.append(
                f"FAIL: core sleeve weight {sleeve_core:.2%} exceeds max {self.max_sleeve_pct:.2%}"
            )

        # 5. Live drawdown circuit breaker against peak equity.
        if live_equity is not None and self.peak_equity is not None and self.peak_equity > 0:
            dd = (live_equity - self.peak_equity) / self.peak_equity * 100.0
            if dd < self.max_drawdown_pct:
                messages.append(
                    f"FAIL: live drawdown {dd:.2f}% exceeds limit {self.max_drawdown_pct:.2f}%"
                )

        # 6. Force paper mode until explicitly cleared.
        paper_mode = os.environ.get("ALPACA_LIVE", "").lower() != "true"
        if not paper_mode:
            messages.append("WARN: running in LIVE mode")

        ok = not any(m.startswith("FAIL:") for m in messages)
        if not ok and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            pd.Series(messages).to_csv(
                self.log_dir / f"risk_block_{today:%Y%m%d}.csv", index=False
            )
        return RiskCheckResult(ok, messages)

    def should_run_today(self, today: Optional[date] = None) -> bool:
        """Skip weekends and obvious US holidays (simple version)."""
        d = today or date.today()
        if d.weekday() >= 5:
            return False
        observed_holidays = {
            (1, 1),   # New Year's
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        }
        return (d.month, d.day) not in observed_holidays
