"""07:30 discretionary morning report.

Computes today's systematic targets, asks the LLM (headless `claude`) for a market
analysis + a recommended profile, builds three deterministic sizing combinations
(aggressive / balanced / passive), renders a LaTeX PDF, and writes a JSON the
runner reads back when the human picks a profile:

    python scripts/rebalance.py --profile <name>

The LLM only advises; all weights are computed deterministically.

Usage:
  cd Live
  python scripts/morning_report.py [--date YYYY-MM-DD] [--no-llm] [--prefer-yfinance]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Repo root (parent of scripts/) on path so `live.*` and `scripts.*` resolve.
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
from live.discretionary import PROFILES, analyze, apply_profile, build_report
from live.portfolio import decompose_target_to_tickers
from scripts.rebalance import compute_systematic_targets


LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="07:30 discretionary morning report (LaTeX PDF)")
    p.add_argument("--date", type=str, default=None, help="Run as-of date (YYYY-MM-DD)")
    p.add_argument("--no-llm", action="store_true", help="Skip the LLM call (deterministic-only)")
    p.add_argument("--prefer-yfinance", action="store_true", help="Use yfinance instead of Alpaca")
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


def _no_analysis() -> dict:
    return {
        "market_analysis": "(LLM analysis skipped via --no-llm.)",
        "recommended_profile": "",
        "confidence": "",
        "rationale": "",
        "headlines": [],
        "source": "none",
    }


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

    profiles = {}
    for name in PROFILES:
        sleeve = apply_profile(name)
        tickers = decompose_target_to_tickers(sleeve, latest_a, latest_b, prices)
        profiles[name] = {"sleeve": sleeve.to_dict(), "tickers": tickers}

    analysis = _no_analysis() if args.no_llm else analyze(run_date)

    record = {
        "date": run_date.isoformat(),
        "equity": equity,
        "systematic": systematic,
        "profiles": profiles,
        "analysis": analysis,
    }
    json_path = LOG_DIR / f"discretionary_{run_date.isoformat()}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    pdf_path = LOG_DIR / f"morning_report_{run_date.isoformat()}.pdf"
    out = build_report(run_date, systematic, profiles, analysis, equity, pdf_path)
    print(f"Wrote {out}")

    rec = analysis.get("recommended_profile", "")
    print(f"Recommended profile: {rec or 'none'} "
          f"(confidence: {analysis.get('confidence', 'n/a')}, "
          f"source: {analysis.get('source')})")
    print("Pick one and run: "
          f"python scripts/rebalance.py --profile {rec or '<name>'} --date {run_date.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())