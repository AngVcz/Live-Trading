"""Risk guardrails for the live strategy runner."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskCheckResult:
    ok: bool
    messages: List[str]


# --- Holiday helpers (NYSE schedule, stdlib only) ---------------------------
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """nth `weekday` (0=Mon..6=Sun) of `month`/`year`."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last `weekday` (0=Mon..6=Sun) of `month`/`year` (month in 1..11)."""
    last = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed(d: date) -> date:
    """NYSE observation rule: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:  # Sat
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sun
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """Anonymous Gregorian computus (matches Python's dateutil.easter)."""
    a = year % 19
    b = year // 100
    c = year % 100
    dd = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - dd - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _market_holidays(year: int) -> set:
    """Set of observed US market holiday dates for `year` (NYSE closures)."""
    fixed = {
        _observed(date(year, 1, 1)),    # New Year's Day
        _observed(date(year, 7, 4)),    # Independence Day
        _observed(date(year, 12, 25)), # Christmas
        _observed(date(year, 6, 19)),  # Juneteenth
    }
    computed = {
        _nth_weekday(year, 1, 0, 3),          # MLK Day: 3rd Monday of Jan
        _last_weekday(year, 5, 0),            # Memorial Day: last Monday of May
        _nth_weekday(year, 9, 0, 1),           # Labor Day: 1st Monday of Sep
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving: 4th Thursday of Nov
        _easter(year) - timedelta(days=2),     # Good Friday
    }
    return fixed | computed


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
        sleeve_weights: Optional[object] = None,
    ) -> RiskCheckResult:
        messages: List[str] = []
        today = current_date or date.today()
        last_price_date = prices.index[-1].date()

        # 1. Stale data check.
        if (today - last_price_date).days > self.stale_data_days:
            messages.append(
                f"FAIL: price data stale (last {last_price_date}, today {today})"
            )

        # 2. Ticker availability + finite, recent last price.
        # ponytail: `t in prices.columns` is false assurance — a failed download
        # leaves an all-NaN column. Require a finite positive latest price AND that
        # the ticker's own last_valid_index is recent vs the panel's last date (so a
        # stale equity target can't hide behind a fresh panel date from another ticker).
        last_price_date = prices.index[-1].date()
        for t, w in target_tickers.items():
            if abs(w) <= 1e-6:
                continue
            if t not in prices.columns:
                messages.append(f"FAIL: target ticker {t} not in price panel")
                continue
            # ponytail: last_valid_index is the source of truth for "latest price" --
            # the last panel row may be NaN (recent gap/delisting) while an earlier
            # bar is valid. A failed download leaves no valid bar at all.
            lvi = prices[t].last_valid_index()
            if lvi is None:
                messages.append(
                    f"FAIL: target ticker {t} has no valid price in panel (all-NaN/failed download)"
                )
                continue
            last = prices[t].loc[lvi]
            if not np.isfinite(last) or last <= 0:
                messages.append(
                    f"FAIL: target ticker {t} latest valid price non-finite/non-positive"
                )
                continue
            if (last_price_date - lvi.date()).days > self.stale_data_days:
                messages.append(
                    f"FAIL: target ticker {t} last valid price stale (last valid {lvi.date()})"
                )

        # 3. Single position limit (ticker-level).
        # ponytail: BIL is the cash proxy (duration ~0, ~T-bill yield), not a risk
        # position. Capping it at 50% blocks the systematic baseline whenever the
        # rates AND cta sleeves both fall back to BIL (0.2 ballast + 0.2 rates->BIL
        # + 0.2 cta->BIL = 0.6) and every risk-off discretionary profile. Exempt BIL
        # only; every real risk position is still capped at max_single_position_pct.
        for t, w in target_tickers.items():
            if t == "BIL":
                continue
            # ponytail: abs() so a short (negative weight) above the cap in magnitude
            # is still blocked — the old `w > max` let shorts bypass it entirely.
            if abs(w) > self.max_single_position_pct:
                messages.append(
                    f"FAIL: position {t} weight {w:.2%} exceeds max {self.max_single_position_pct:.2%}"
                )

        # 4. Sleeve concentration (A+B combined should not dominate).
        # Use SLEEVE-level weights when available; the old ticker-keyed lookup
        # (`target_tickers.get("A") + target_tickers.get("B")`) was always 0 because
        # target_tickers is keyed by ticker (e.g. "SPY"), not by sleeve name.
        if sleeve_weights is not None:
            sw = sleeve_weights.to_dict() if hasattr(sleeve_weights, "to_dict") else dict(sleeve_weights)
            sleeve_core = float(sw.get("A", 0.0) + sw.get("B", 0.0))
        else:
            # ponytail: fallback when no sleeve weights passed (always 0 for ticker-keyed dict).
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
        """Skip weekends and major US market holidays (NYSE schedule)."""
        d = today or date.today()
        if d.weekday() >= 5:
            return False
        return d not in _market_holidays(d.year)
