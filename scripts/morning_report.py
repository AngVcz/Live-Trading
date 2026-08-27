"""07:30 discretionary morning report (two-stage news overlay).

Pipeline: systematic targets -> deterministic metrics panel -> Stage-1 LLM
exec summary (prompts/01) -> deterministic tilt options from today's sizings ->
Stage-2 LLM committee decision (prompts/02) -> LaTeX PDF + JSON. The user then
trades one option:  python scripts/rebalance.py --option <systematic|risk_on|risk_off>

The LLM never emits weights; every weight in the report is deterministic.

Usage:
  cd Live
  python scripts/morning_report.py [--date YYYY-MM-DD] [--no-llm] [--prefer-yfinance]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
except Exception as e:
    print(f"WARN: could not load .env: {e}")

from live.data_feed import get_last_trading_day
from live.discretionary import (
    _no_stage1,
    _no_stage2,
    build_report,
    build_tilt_options,
    stage1_summarize,
    stage2_decide,
)
from live.morning_metrics import compute_metrics_panel
from scripts.rebalance import compute_systematic_targets

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="07:30 discretionary morning report")
    p.add_argument("--date", type=str, default=None, help="Run as-of date (YYYY-MM-DD)")
    p.add_argument("--no-llm", action="store_true", help="Skip both LLM stages")
    p.add_argument("--prefer-yfinance", action="store_true", help="Use yfinance instead of Alpaca")
    p.add_argument("--model", type=str, default=None,
                   help="Override the claude CLI model for the LLM stages (e.g. claude-fable-5)")
    return p.parse_args()


def _get_equity() -> float:
    """Read paper account equity if credentials are present, else 100,000."""
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET")):
        return 100_000.0
    try:
        from live.alpaca_executor import AlpacaExecutor
        return float(AlpacaExecutor().get_account()["equity"])
    except Exception as e:
        print(f"WARN: could not read account equity ({e}); using 100,000")
        return 100_000.0


def main() -> int:
    args = _parse_args()
    run_date = date.fromisoformat(args.date) if args.date else get_last_trading_day()
    prefer_alpaca = not args.prefer_yfinance

    print(f"[{datetime.now()}] Morning report for {run_date} (no_llm={args.no_llm})")

    try:
        sys_targets = compute_systematic_targets(run_date, prefer_alpaca=prefer_alpaca)
    except Exception as e:
        print(f"FATAL: systematic pipeline failed: {e}")
        return 1

    prices = sys_targets["prices"]
    latest_a = sys_targets["weights_a_last"]
    latest_b = sys_targets["weights_b_last"]
    equity = _get_equity()

    systematic = {
        "sleeve": sys_targets["sleeve_weights"].to_dict(),
        "tickers": sys_targets["ticker_weights"],
    }

    metrics = compute_metrics_panel(
        prices, sys_targets["ticker_weights"],
        equity=equity, as_of=run_date)
    metrics["sleeve_weights"] = systematic["sleeve"]

    options = build_tilt_options(
        sys_targets["sleeve_weights"], latest_a, latest_b, prices,
        vix_overlay_active=(metrics.get("vix_overlay_active") is True))

    if args.no_llm:
        stage1 = _no_stage1("skipped via --no-llm")
        stage2 = _no_stage2("skipped via --no-llm")
    else:
        stage1 = stage1_summarize(run_date, metrics, model=args.model)
        metrics["macro_calendar"] = stage1.get("macro_calendar", [])
        # Stage 2 always runs (its veto list covers a failed Stage 1).
        stage2 = stage2_decide(run_date, stage1, options, model=args.model)

    record = {
        "date": run_date.isoformat(),
        "equity": equity,
        "systematic": systematic,
        "metrics": metrics,
        "options": options,
        "stage1": stage1,
        "stage2": stage2,
    }
    json_path = LOG_DIR / f"discretionary_{run_date.isoformat()}.json"
    json_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path}")

    pdf_path = LOG_DIR / f"morning_report_{run_date.isoformat()}.pdf"
    out = build_report(run_date, systematic, metrics, options, stage1, stage2,
                       equity, pdf_path)
    print(f"Wrote {out}")

    rec = stage2.get("recommended_option", "")
    if stage2.get("veto") == "yes":
        rec = "systematic"
    print(f"Stage 1 regime bias: {stage1.get('regime_bias') or 'n/a'} "
          f"(confidence: {stage1.get('summary_confidence') or 'n/a'}, "
          f"source: {stage1.get('source')})")
    print(f"Committee recommendation: {rec or 'none'} "
          f"(confidence: {stage2.get('confidence') or 'n/a'}, "
          f"veto: {stage2.get('veto') or 'n/a'}, source: {stage2.get('source')})")
    print("Pick one and run: "
          f"python scripts/rebalance.py --option {rec or '<name>'} --date {run_date.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())