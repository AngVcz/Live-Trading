"""Persistent state (peak equity, last weights) for the live runner."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent  # live/ -> repo root
STATE_PATH = REPO_ROOT / "logs" / "state.json"


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def update_peak_equity(equity: float) -> float:
    state = load_state()
    peak = max(state.get("peak_equity", 0.0), equity)
    state["peak_equity"] = peak
    save_state(state)
    return peak


def get_peak_equity() -> Optional[float]:
    return load_state().get("peak_equity")


def load_last_weights() -> Optional[pd.Series]:
    state = load_state()
    weights = state.get("last_weights")
    if weights is None:
        return None
    return pd.Series(weights)


def save_last_weights(weights: Dict[str, float], as_of: date) -> None:
    state = load_state()
    state["last_weights"] = weights
    state["last_weights_date"] = as_of.isoformat()
    save_state(state)
