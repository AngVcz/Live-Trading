"""Lightweight live vs backtest monitoring for the A+B+Diversifier Sleeves strategy.

Reads the JSONL run log produced by ``live_runner.py`` and compares live PnL,
sleeve weights, and turnover against the historical backtest expectation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Repo root (parent of live/) — must match scripts/rebalance.py LOG_DIR, which writes
# ``logs/target_weights.jsonl``. The old path pointed one level too high
# (Trading I/.claude/cache/live/...) so the monitor always saw "no live records yet".
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "logs" / "target_weights.jsonl"


def load_run_log(path: Optional[Path] = None) -> pd.DataFrame:
    """Load all daily run records into a DataFrame."""
    path = path or LOG_PATH
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


def sleeve_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticker targets into sleeve-level weights per day."""
    if df.empty or "target_weights" not in df.columns:
        return pd.DataFrame()

    sleeve_map = {
        "A": ["SPY", "QQQ", "IWM", "VTI", "VXUS", "XLK", "XLV", "XLI", "XLF", "XLE",
              "XLU", "XLP", "XLY", "XLB", "XLRE", "AAPL", "MSFT", "AMZN", "GOOGL",
              "NVDA", "META", "TSLA", "JPM", "BTC-USD", "ETH-USD"],
        "rates": ["TLT", "IEF", "BIL"],
        "bear": ["SH", "BIL"],
        "cta": ["PDBC", "DBMF", "KMLM", "BIL"],
    }

    records: List[Dict[str, float]] = []
    for date, row in df.iterrows():
        targets = row["target_weights"]
        rec: Dict[str, float] = {"date": date}
        for sleeve, tickers in sleeve_map.items():
            rec[sleeve] = sum(targets.get(t, 0.0) for t in tickers)
        records.append(rec)
    out = pd.DataFrame(records).set_index("date")
    return out


def summary(path: Optional[Path] = None) -> Dict[str, Any]:
    df = load_run_log(path)
    if df.empty:
        return {"status": "no live records yet"}

    latest = df.iloc[-1]
    orders = latest.get("orders", [])
    equity = latest["account"]["equity"]
    turnover = sum(o["notional"] for o in orders) / equity if equity else 0.0

    return {
        "latest_date": latest.name.strftime("%Y-%m-%d") if hasattr(latest, "name") else None,
        "dry_run": latest.get("dry_run", True),
        "account_equity": equity,
        "num_orders_today": len(orders),
        "turnover_pct": turnover * 100,
        "sleeve_weights": sleeve_attribution(df).iloc[-1].to_dict() if not sleeve_attribution(df).empty else {},
    }


if __name__ == "__main__":
    from pprint import pprint
    pprint(summary())
