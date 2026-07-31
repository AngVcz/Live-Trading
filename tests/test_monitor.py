"""Smoke tests for live/monitor.py LOG_PATH + load_run_log.

The monitor used to point one level too high (Trading I/.claude/cache/live/...),
so it always reported "no live records yet" even after a real rebalance run.
These tests pin LOG_PATH to the same path scripts/rebalance.py writes
(``logs/target_weights.jsonl`` under the Live repo root) and prove load_run_log
parses a real JSONL record.
"""
from pathlib import Path

import pandas as pd

from live import monitor


def test_log_path_matches_rebalance_weight_log():
    """monitor.LOG_PATH must equal rebalance.WEIGHT_LOG (the writer's path)."""
    import scripts.rebalance as rebalance  # import triggers its sys.path/.env setup; no orders

    assert monitor.LOG_PATH == rebalance.WEIGHT_LOG, (
        f"monitor {monitor.LOG_PATH} != rebalance {rebalance.WEIGHT_LOG}"
    )


def test_log_path_is_under_live_repo_logs():
    """LOG_PATH must live under <Live>/logs, not under Trading I/.claude/..."""
    s = str(monitor.LOG_PATH).replace("\\", "/")
    assert s.endswith("/logs/target_weights.jsonl"), s
    assert "/.claude/cache/" not in s, s  # the old (wrong) location


def test_load_run_log_parses_a_record(tmp_path: Path):
    """A real JSONL record round-trips through load_run_log."""
    p = tmp_path / "target_weights.jsonl"
    p.write_text(
        '{"date": "2026-07-30", "dry_run": false, '
        '"account": {"equity": 100000.0}, '
        '"target_weights": {"SPY": 0.20, "TLT": 0.20, "BIL": 0.60}, '
        '"orders": [{"notional": 20000.0}]}\n',
        encoding="utf-8",
    )
    df = monitor.load_run_log(p)
    assert list(df.index) == [pd.Timestamp("2026-07-30")]
    assert not df.iloc[-1]["dry_run"]  # numpy bool -> truthiness, not identity


def test_summary_returns_no_records_when_missing(tmp_path: Path):
    """summary() on a missing log returns the documented 'no live records yet'."""
    s = monitor.summary(tmp_path / "missing.jsonl")
    assert s == {"status": "no live records yet"}