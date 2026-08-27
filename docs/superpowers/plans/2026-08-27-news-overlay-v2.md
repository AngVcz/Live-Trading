# News Overlay v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Live 07:30 morning report to a two-stage LLM news overlay (exec summary → committee decision) with a deterministic metrics panel and three options built by tilting today's actual sizings.

**Architecture:** All weights stay deterministic Python (`build_tilt_options` in `live/discretionary.py`, metrics in new `live/morning_metrics.py`); two headless `claude` calls driven by SOPs in `Live/prompts/` emit only prose + control-line labels. Spec: `docs/superpowers/specs/2026-08-26-news-overlay-v2-design.md`.

**Tech Stack:** Python 3.13, pandas, scipy, yfinance (cached), headless `claude` CLI (WebSearch), MiKTeX pdflatex, pytest.

**Repo/branch:** work in `Live/` (its own git repo), branch `feat/news-exec-summary`. All commands run from `Live/`. Commit trailer on every commit: `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

## Task 1: Tilt engine in `live/discretionary.py`

**Files:**
- Modify: `live/discretionary.py` (add; do NOT delete `PROFILES`/`apply_profile` yet — rebalance.py still imports them until Task 6)
- Test: `tests/test_live_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_live_pipeline.py`:

```python
def test_tilt_options_bounds():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns
    from live.discretionary import build_tilt_options, OPTION_NAMES
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    options = build_tilt_options(sleeve, weights_a.iloc[-1], weights_b.iloc[-1], prices)
    assert set(options) == set(OPTION_NAMES)
    ab = {}
    for name in OPTION_NAMES:
        o = options[name]
        assert abs(sum(o["tickers"].values()) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in o["tickers"].values())
        assert all(w <= 0.35 + 1e-9 for t, w in o["tickers"].items() if t != "BIL")
        for k in ("A", "B", "rates", "cta"):
            assert o["sleeve"][k] <= 0.45 + 1e-9, (name, k)
        ab[name] = o["sleeve"]["A"] + o["sleeve"]["B"]
    assert ab["risk_off"] < ab["systematic"] < ab["risk_on"]
    assert ab["risk_on"] <= 0.70 + 1e-9


def test_tilt_options_vix_overlay_disables_risk_on():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns
    from live.discretionary import build_tilt_options
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    options = build_tilt_options(sleeve, weights_a.iloc[-1], weights_b.iloc[-1],
                                 prices, vix_overlay_active=True)
    diff = sum(abs(options["risk_on"]["sleeve"][k] - options["systematic"]["sleeve"][k])
               for k in options["systematic"]["sleeve"])
    assert diff < 1e-12
    assert "disabled" in options["risk_on"]["note"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_live_pipeline.py::test_tilt_options_bounds -q`
Expected: FAIL — `ImportError: cannot import name 'build_tilt_options'`

- [ ] **Step 3: Implement the tilt engine** — in `live/discretionary.py`, add after the imports (before `PROFILES`):

```python
PROMPTS_DIR = REPO_ROOT / "prompts"
SLEEVE_ORDER = ["A", "B", "rates", "BIL_ballast", "cta"]
_SLEEVE_ORDER = SLEEVE_ORDER  # alias; removed in Task 6 with apply_profile
OPTION_NAMES = ("systematic", "risk_on", "risk_off")
_RISK_SLEEVES = ("A", "B", "rates", "cta")   # BIL_ballast is the cash sink, uncapped
SLEEVE_CAP = 0.45
AB_CAP = 0.70
TICKER_CAP = 0.35                          # margin below the 0.50 risk-guardrail line
_RISK_ON_MULT = {"A": 1.5, "B": 1.5, "cta": 1.5}
_RISK_OFF_MULT = {"A": 0.5, "B": 0.5, "cta": 0.5}


def _base_sleeve(sleeve: pd.Series) -> pd.Series:
    """Reindex to the five live sleeves ('bear' is 0 by default) as floats."""
    return sleeve.reindex(SLEEVE_ORDER).fillna(0.0).astype(float)


def _join_note(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _enforce_caps(s: pd.Series) -> tuple[pd.Series, str]:
    """Repair cap breaches by spilling the excess into BIL_ballast."""
    s = s.copy()
    notes = []
    ab = s["A"] + s["B"]
    if ab > AB_CAP:
        f = AB_CAP / ab
        s["BIL_ballast"] += (s["A"] - s["A"] * f) + (s["B"] - s["B"] * f)
        s["A"] *= f
        s["B"] *= f
        notes.append("A+B capped at 70%")
    for k in _RISK_SLEEVES:
        if s[k] > SLEEVE_CAP:
            s["BIL_ballast"] += s[k] - SLEEVE_CAP
            s[k] = SLEEVE_CAP
            notes.append(f"{k} capped at 45%")
    return s, _join_note(*notes)


def _tilt_risk_on(s: pd.Series) -> tuple[pd.Series, str]:
    out = s.copy()
    for k, m in _RISK_ON_MULT.items():
        out[k] = s[k] * m
    extra = float(sum(out[k] - s[k] for k in _RISK_ON_MULT))
    for src in ("BIL_ballast", "rates"):  # fund from cash first, then rates
        take = min(out[src], extra)
        out[src] -= take
        extra -= take
    if extra > 1e-9:
        return s.copy(), "risk-on tilt unfundable; systematic kept"
    return out, ""


def _tilt_risk_off(s: pd.Series, rates_in_uptrend: bool) -> tuple[pd.Series, str]:
    out = s.copy()
    for k, m in _RISK_OFF_MULT.items():
        out[k] = s[k] * m
    freed = float(sum(s[k] - out[k] for k in _RISK_OFF_MULT))
    if rates_in_uptrend:
        to_rates = min(freed, max(0.0, SLEEVE_CAP - out["rates"]))
        out["rates"] += to_rates
        freed -= to_rates
    out["BIL_ballast"] += freed
    return out, ""


def _clip_tickers(tickers: Dict[str, float]) -> Dict[str, float]:
    out = dict(tickers)
    for t, w in list(out.items()):
        if t != "BIL" and w > TICKER_CAP:
            out[t] = TICKER_CAP
            out["BIL"] = out.get("BIL", 0.0) + (w - TICKER_CAP)
    return out


def _rates_in_uptrend(prices: pd.DataFrame) -> bool:
    """True when TLT or IEF closed above its 200-day SMA on the last bar."""
    for t in ("TLT", "IEF"):
        if t in prices.columns:
            col = prices[t].dropna()
            if len(col) >= 200 and col.iloc[-1] > col.tail(200).mean():
                return True
    return False


def build_tilt_options(
    sleeve_weights: pd.Series,
    weights_a: pd.Series,
    weights_b: pd.Series,
    prices: pd.DataFrame,
    vix_overlay_active: bool = False,
) -> Dict[str, Dict]:
    """Build the three report options by tilting TODAY's systematic sizings.

    Pure and deterministic: no fetch, no LLM. Each option is
    {"sleeve": {...}, "tickers": {...}, "note": str} with weights summing to 1.0.
    """
    from live.portfolio import decompose_target_to_tickers

    base = _base_sleeve(sleeve_weights)
    variants = {
        "systematic": (base.copy(), ""),
        "risk_on": _tilt_risk_on(base),
        "risk_off": _tilt_risk_off(base, _rates_in_uptrend(prices)),
    }
    options: Dict[str, Dict] = {}
    for name in OPTION_NAMES:
        sv, note = variants[name]
        if name == "risk_on" and vix_overlay_active:
            sv, note = base.copy(), "risk-on tilt disabled by active VIX overlay"
        sv, cap_note = _enforce_caps(sv)
        total = float(sv.sum())
        if total > 0:
            sv = sv / total
        tickers = _clip_tickers(decompose_target_to_tickers(sv, weights_a, weights_b, prices))
        options[name] = {"sleeve": {k: float(v) for k, v in sv.items()},
                         "tickers": tickers,
                         "note": _join_note(note, cap_note)}
    return options
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_live_pipeline.py::test_tilt_options_bounds tests/test_live_pipeline.py::test_tilt_options_vix_overlay_disables_risk_on -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add live/discretionary.py tests/test_live_pipeline.py
git commit -m "feat(discretionary): add deterministic tilt engine (systematic/risk_on/risk_off from today's sizings)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Metrics panel — `live/morning_metrics.py`

**Files:**
- Create: `live/morning_metrics.py`
- Test: `tests/test_live_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_live_pipeline.py`:

```python
def test_metrics_panel_smoke():
    from live.core_signals import build_core_returns
    from live.portfolio import SleeveConfig, build_live_weights, build_sleeve_returns, decompose_target_to_tickers
    from live.morning_metrics import compute_metrics_panel
    prices = _load_prices().rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, weights_a, weights_b = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    tickers = decompose_target_to_tickers(sleeve, weights_a.iloc[-1], weights_b.iloc[-1], prices)
    m = compute_metrics_panel(prices, tickers, weights_a.iloc[-1], weights_b.iloc[-1],
                              equity=100_000.0, as_of=date(2025, 6, 30))
    for key in ("as_of", "vix_close", "vix_change_1d", "vix_percentile_252d",
                "vix_overlay_active", "tnx_10y_level", "tnx_change_5d",
                "tlt_above_sma200", "ief_above_sma200", "breadth_pct_above_sma200",
                "holdings_below_sma200", "equity", "peak_equity", "drawdown_pct",
                "guardrail_margin_pct", "turnover_oneway_pct", "sleeve_weights",
                "top_tickers", "macro_calendar"):
        assert key in m, key
    assert isinstance(m["vix_overlay_active"], bool)
    if m["vix_percentile_252d"] != "n/a":
        assert 0.0 <= m["vix_percentile_252d"] <= 1.0
    assert isinstance(m["holdings_below_sma200"], list)
    assert m["macro_calendar"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_pipeline.py::test_metrics_panel_smoke -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'live.morning_metrics'`

- [ ] **Step 3: Write `live/morning_metrics.py`**

```python
"""Deterministic metrics panel for the 07:30 morning report.

Every number the LLM sees is computed here from the price panel and persisted
state — the model interprets these values, it never invents them. Any single
metric that cannot be computed becomes the string "n/a"; the panel as a whole
never raises.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

import pandas as pd
from scipy import stats

from live.core_signals import UNIVERSE as CORE_UNIVERSE
from live.state import get_peak_equity, load_last_weights

_NA = "n/a"


def _last_two(col: pd.Series):
    s = col.dropna()
    return float(s.iloc[-1]), float(s.iloc[-2])


def _above_sma200(col: pd.Series) -> bool:
    s = col.dropna()
    if len(s) < 200:
        raise ValueError("short history")
    return bool(s.iloc[-1] > s.tail(200).mean())


def _vix_metrics(prices: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    col = prices["VIX"] if "VIX" in prices.columns else prices["^VIX"]
    last, prev = _last_two(col)
    out["vix_close"] = round(last, 2)
    out["vix_change_1d"] = round(last - prev, 2)
    window = col.dropna().tail(252)
    if len(window) < 126:
        raise ValueError("short VIX history")
    pct = float(stats.percentileofscore(window, last, kind="rank") / 100.0)
    out["vix_percentile_252d"] = round(pct, 3)
    out["vix_overlay_active"] = pct > 0.70   # VIX_OVERLAY_PCTILE in core_signals
    return out


def _tnx_metrics(as_of: date) -> Dict[str, Any]:
    from live.data_feed import fetch_panel
    # ^TNX is not Alpaca-tradeable -> always yfinance (cached).
    panel = fetch_panel(["^TNX"], as_of - timedelta(days=365), as_of,
                        prefer_alpaca=False)
    col = panel["^TNX"].dropna()
    if len(col) < 6:
        raise ValueError("short TNX history")
    return {
        # ^TNX quotes the 10Y yield x 10 (CBOE convention).
        "tnx_10y_level": round(float(col.iloc[-1]) / 10.0, 3),
        "tnx_change_5d": round(float(col.iloc[-1] - col.iloc[-6]) / 10.0, 3),
    }


def _breadth(prices: pd.DataFrame) -> Dict[str, Any]:
    cols = [t for t in CORE_UNIVERSE if t in prices.columns
            and prices[t].dropna().shape[0] >= 200]
    if not cols:
        raise ValueError("no universe columns")
    above = sum(1 for t in cols if _above_sma200(prices[t]))
    return {"breadth_pct_above_sma200": round(100.0 * above / len(cols), 1),
            "breadth_universe_n": len(cols)}


def _holdings_below(prices: pd.DataFrame, weights_a: pd.Series, weights_b: pd.Series):
    held = sorted(set(weights_a[weights_a > 0].index) | set(weights_b[weights_b > 0].index))
    below = []
    for t in held:
        if t in prices.columns:
            try:
                if not _above_sma200(prices[t]):
                    below.append(t)
            except ValueError:
                pass
    return below


def _book_metrics(equity: float) -> Dict[str, Any]:
    peak = get_peak_equity() or equity
    dd = (equity - peak) / peak if peak > 0 else 0.0
    return {"peak_equity": round(float(peak), 2),
            "drawdown_pct": round(100.0 * dd, 2),
            "guardrail_margin_pct": round(100.0 * (equity - 0.9 * peak) / peak, 2)
            if peak > 0 else _NA}


def _turnover(ticker_weights: Dict[str, float]):
    last = load_last_weights()
    if last is None:
        raise ValueError("no last weights")
    lw = last.to_dict()
    tickers = set(ticker_weights) | set(lw)
    tw = sum(abs(ticker_weights.get(t, 0.0) - lw.get(t, 0.0)) for t in tickers) / 2.0
    return round(100.0 * tw, 2)


def compute_metrics_panel(
    prices: pd.DataFrame,
    ticker_weights: Dict[str, float],
    weights_a: pd.Series,
    weights_b: pd.Series,
    equity: float,
    as_of: date,
) -> Dict[str, Any]:
    """The full daily metrics panel. Values are exact; "n/a" marks a failed cell."""
    m: Dict[str, Any] = {"as_of": as_of.isoformat(), "equity": round(float(equity), 2)}
    for block in (
        lambda: _vix_metrics(prices),
        lambda: _tnx_metrics(as_of),
        lambda: {"tlt_above_sma200": _above_sma200(prices["TLT"])},
        lambda: {"ief_above_sma200": _above_sma200(prices["IEF"])},
        lambda: _breadth(prices),
        lambda: {"holdings_below_sma200": _holdings_below(prices, weights_a, weights_b)},
        lambda: _book_metrics(equity),
        lambda: {"turnover_oneway_pct": _turnover(ticker_weights)},
    ):
        try:
            m.update(block())
        except Exception:
            pass  # a failed block leaves its keys at the setdefault 'n/a' below
    m.setdefault("vix_close", _NA)
    m.setdefault("vix_change_1d", _NA)
    m.setdefault("vix_percentile_252d", _NA)
    m.setdefault("vix_overlay_active", False)
    m.setdefault("tnx_10y_level", _NA)
    m.setdefault("tnx_change_5d", _NA)
    m.setdefault("tlt_above_sma200", _NA)
    m.setdefault("ief_above_sma200", _NA)
    m.setdefault("breadth_pct_above_sma200", _NA)
    m.setdefault("holdings_below_sma200", _NA)
    m.setdefault("peak_equity", _NA)
    m.setdefault("drawdown_pct", _NA)
    m.setdefault("guardrail_margin_pct", _NA)
    m.setdefault("turnover_oneway_pct", _NA)
    m["sleeve_weights"] = {}  # filled by caller (kept with the prompt payload)
    m["top_tickers"] = dict(sorted(ticker_weights.items(), key=lambda kv: kv[1],
                                   reverse=True)[:10])
    m["macro_calendar"] = []  # Stage 1 (LLM) fills today's events from the web
    return m


def _self_check() -> None:
    """Compute the panel on a 4-year cached panel; assert keys, types, ranges."""
    from live.core_signals import build_core_returns
    from live.data_feed import fetch_panel
    from live.portfolio import (SleeveConfig, build_live_weights,
                                build_sleeve_returns, decompose_target_to_tickers)
    end = date(2025, 6, 30)
    tickers = list(dict.fromkeys(
        CORE_UNIVERSE + ["TLT", "IEF", "PDBC", "KMLM", "DBMF", "BIL", "^VIX"]))
    prices = fetch_panel(tickers, end - timedelta(days=365 * 4), end,
                         prefer_alpaca=False).rename(columns={"^VIX": "VIX"})
    ret_a, ret_b, wa, wb = build_core_returns(prices, commission_bps=10.0)
    sleeve = build_live_weights(ret_a, ret_b, build_sleeve_returns(prices), SleeveConfig())
    tw = decompose_target_to_tickers(sleeve, wa.iloc[-1], wb.iloc[-1], prices)
    m = compute_metrics_panel(prices, tw, wa.iloc[-1], wb.iloc[-1],
                              equity=100_000.0, as_of=end)
    assert isinstance(m["vix_overlay_active"], bool)
    if m["vix_percentile_252d"] != _NA:
        assert 0.0 <= m["vix_percentile_252d"] <= 1.0
    assert isinstance(m["holdings_below_sma200"], list)
    assert m["macro_calendar"] == []
    print(f"morning_metrics self-check OK: vix={m['vix_close']} "
          f"pctile={m['vix_percentile_252d']} breadth={m['breadth_pct_above_sma200']}%")


if __name__ == "__main__":
    _self_check()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_live_pipeline.py::test_metrics_panel_smoke -q`
Expected: PASS. Also run `python -m live.morning_metrics` → expect `morning_metrics self-check OK: ...`

- [ ] **Step 5: Commit**

```bash
git add live/morning_metrics.py tests/test_live_pipeline.py
git commit -m "feat(report): add deterministic morning metrics panel (VIX pctile, TNX, breadth, book, turnover)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Stage parsers + LLM plumbing in `live/discretionary.py`

**Files:**
- Modify: `live/discretionary.py` (add `_call_claude`, `stage1_summarize`, `stage2_decide`, `_parse_stage1`, `_parse_stage2`; keep old `analyze()` until Task 5)
- Test: `tests/test_live_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_live_pipeline.py`:

```python
def test_stage_parsers():
    from live.discretionary import _parse_stage1, _parse_stage2
    s1 = _parse_stage1(
        "- Fed held rates steady.\n- CPI cooled to 2.9% y/y.\n\n"
        "## Macro regime\nDisinflation continues; breadth improving.\n\n"
        "## Today's events\n- 14:30 ET: FOMC minutes.\n\n"
        "## Metrics read\nVIX percentile near mid-range.\n\n"
        "REGIME_BIAS: neutral\nSUMMARY_CONFIDENCE: med\n")
    assert s1["regime_bias"] == "neutral"
    assert s1["summary_confidence"] == "med"
    assert "FOMC" in " ".join(s1["macro_calendar"])
    assert "REGIME_BIAS" not in s1["exec_summary"]
    bad1 = _parse_stage1("text\nREGIME_BIAS: euphoric\nSUMMARY_CONFIDENCE: max\n")
    assert bad1["regime_bias"] == "" and bad1["summary_confidence"] == ""

    s2 = _parse_stage2(
        "## Assessment\nBreadth supports risk, but the FOMC is a two-way event.\n\n"
        "## Option ranking\n1. systematic — event day, stand pat.\n"
        "2. risk_off — cheap insurance.\n3. risk_on — premature.\n\n"
        "RECOMMENDED_OPTION: systematic\nCONFIDENCE: med\nVETO: no\n")
    assert s2["recommended_option"] == "systematic"
    assert s2["confidence"] == "med" and s2["veto"] == "no"
    assert s2["ranking"][0] == {"rank": 1, "option": "systematic"}
    assert "RECOMMENDED_OPTION" not in s2["assessment"]
    bad2 = _parse_stage2("text\nRECOMMENDED_OPTION: yolo\nCONFIDENCE: med\nVETO: maybe\n")
    assert bad2["recommended_option"] == "" and bad2["veto"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_pipeline.py::test_stage_parsers -q`
Expected: FAIL — `ImportError: cannot import name '_parse_stage1'`

- [ ] **Step 3: Implement parsers + stages** — add to `live/discretionary.py` after the tilt engine:

```python
# ---- LLM stage plumbing -----------------------------------------------------

_REGIME_BIAS = ("risk_on", "neutral", "risk_off")
_CONF = ("low", "med", "high")


def _call_claude(prompt: str, model: str | None = None) -> str:
    """One headless ``claude`` CLI call; returns result text or raises RuntimeError."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "WebSearch"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}")
    env = json.loads(proc.stdout)
    if env.get("is_error") or env.get("subtype") != "success":
        raise RuntimeError(env.get("result") or "claude error")
    return env.get("result", "")


def _control_line(text: str, key: str, allowed: tuple[str, ...]) -> str:
    m = re.search(rf"(?im)^\s*{key}:\s*([a-zA-Z_]+)\s*$", text)
    if not m:
        return ""
    val = m.group(1).strip().lower()
    return val if val in allowed else ""


def _strip_control_lines(text: str, keys: tuple[str, ...]) -> str:
    body = text
    for k in keys:
        body = re.sub(rf"(?im)^\s*{k}:.*$\n?", "", body)
    return body.strip()


def _section(text: str, header: str) -> str:
    """Return the text under a '## <header>' line, up to the next '## ' or EOF."""
    m = re.search(rf"(?ims)^##\s+{re.escape(header)}\s*$\n?(.*?)(?=^##\s|\Z)", text)
    return m.group(1).strip() if m else ""


def _parse_stage1(text: str) -> Dict:
    return {
        "exec_summary": _strip_control_lines(text, ("REGIME_BIAS", "SUMMARY_CONFIDENCE")),
        "headlines": re.findall(r"(?m)^\s*[-*•]\s+(.+?)\s*$",
                                _section(text, "Headlines"))[:5],
        "macro_calendar": re.findall(r"(?m)^\s*[-*•]\s+(.+?)\s*$",
                                     _section(text, "Today's events")),
        "regime_bias": _control_line(text, "REGIME_BIAS", _REGIME_BIAS),
        "summary_confidence": _control_line(text, "SUMMARY_CONFIDENCE", _CONF),
    }


def _parse_stage2(text: str) -> Dict:
    ranking = []
    for m in re.finditer(r"(?m)^\s*([123])[\.)]\s*(\w+)\s*[—–-]\s*(.+?)\s*$",
                         _section(text, "Option ranking")):
        opt = m.group(2).strip().lower()
        ranking.append({"rank": int(m.group(1)),
                        "option": opt if opt in OPTION_NAMES else "",
                        "reason": m.group(3).strip()})
    return {
        "assessment": _strip_control_lines(
            text, ("RECOMMENDED_OPTION", "CONFIDENCE", "VETO")),
        "ranking": ranking,
        "recommended_option": _control_line(text, "RECOMMENDED_OPTION", OPTION_NAMES),
        "confidence": _control_line(text, "CONFIDENCE", _CONF),
        "veto": _control_line(text, "VETO", ("yes", "no")),
    }


def _render_prompt(template_name: str, **tokens: str) -> str:
    """Read prompts/<template_name> and substitute ALL-CAPS tokens (no .format:
    the SOPs may contain literal braces in JSON examples)."""
    t = (PROMPTS_DIR / template_name).read_text(encoding="utf-8")
    for k, v in tokens.items():
        t = t.replace("{" + k + "}", v)
    return t


def _no_stage1(reason: str) -> Dict:
    return {"exec_summary": f"(Executive summary unavailable: {reason}.)",
            "headlines": [], "macro_calendar": [], "regime_bias": "",
            "summary_confidence": "", "source": "none"}


def _no_stage2(reason: str) -> Dict:
    return {"assessment": f"(Committee analysis unavailable: {reason}. "
                          "Options tables above are still valid.)",
            "ranking": [], "recommended_option": "", "confidence": "",
            "veto": "", "source": "none"}


def stage1_summarize(run_date: date, metrics: Dict, model: str | None = None) -> Dict:
    """Stage 1: news scrape + executive summary per prompts/01_news_exec_summary.md."""
    prompt = _render_prompt("01_news_exec_summary.md",
                            DATE=run_date.isoformat(),
                            METRICS_JSON=json.dumps(metrics, indent=2, default=str))
    try:
        parsed = _parse_stage1(_call_claude(prompt, model))
        parsed["source"] = "claude-cli"
        return parsed
    except Exception as e:  # ponytail: any failure -> deterministic-only report
        return _no_stage1(str(e))


def stage2_decide(run_date: date, stage1: Dict, options: Dict[str, Dict],
                  model: str | None = None) -> Dict:
    """Stage 2: committee ranking of the three options per prompts/02_options_analysis.md."""
    prompt = _render_prompt("02_options_analysis.md",
                            DATE=run_date.isoformat(),
                            STAGE1_TEXT=stage1.get("exec_summary", "(unavailable)"),
                            OPTIONS_JSON=json.dumps(options, indent=2, default=str))
    try:
        parsed = _parse_stage2(_call_claude(prompt, model))
        parsed["source"] = "claude-cli"
        return parsed
    except Exception as e:
        return _no_stage2(str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_live_pipeline.py::test_stage_parsers -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add live/discretionary.py tests/test_live_pipeline.py
git commit -m "feat(discretionary): add two-stage LLM plumbing + control-line parsers

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Prompt SOPs — `Live/prompts/`

**Files:**
- Create: `prompts/01_news_exec_summary.md`
- Create: `prompts/02_options_analysis.md`

- [ ] **Step 1: Write `prompts/01_news_exec_summary.md`**

````markdown
# Stage 1 — News Scraping → Executive Summary

You are the senior macro strategist at a macro asset-management desk. Today is {DATE}.

The desk runs a daily-rebalanced ETF portfolio with five sleeves: two equity
dual-momentum engines (A and B), a rates sleeve (TLT/IEF/BIL by 200-day trend), a
BIL ballast sleeve (cash proxy), and a CTA proxy (PDBC/DBMF/KMLM in uptrend, else
BIL). A VIX-70th-percentile overlay halves equity exposure when active.

## Your task

1. Use web search to gather the latest financial and macroeconomic news: central
   banks (Fed/ECB/BoJ), inflation (CPI/PCE), employment, GDP, geopolitics,
   Treasury yields, VIX, oil/gold, and any news on the current top holdings in
   the metrics JSON below. Then list **today's** macro calendar (data releases,
   FOMC/speakers, auctions) with ET times.
2. Read the metrics panel — it is exact and authoritative. Interpret it; do not
   restate numbers it does not contain and do not add numbers without naming
   the source.
3. Write the executive summary. Max ~400 words. Every news claim names its
   source (e.g. "per Reuters", "per Bloomberg", "per the BLS release").

## Portfolio metrics (authoritative)

```json
{METRICS_JSON}
```

## Output format — follow exactly

## Headlines
- up to 5 one-line bullets (prefix each with "- ")

## Macro regime
2-4 sentences: where we are (expansion/slowdown, easing/tightening, risk appetite),
with the strongest evidence for and against.

## Today's events
- one bullet per scheduled event, ET time first ("- 08:30 ET: CPI (BLS)")
- "none scheduled" if the calendar is empty

## Metrics read
3-5 sentences interpreting the panel: VIX regime and overlay status, yield level
and trend, breadth, drawdown vs the -10% guardrail, and what today's systematic
turnover implies.

End your response with exactly these two lines and nothing after them:

REGIME_BIAS: <risk_on|neutral|risk_off>
SUMMARY_CONFIDENCE: <low|med|high>

You may NOT propose portfolio weights, sizings, or sleeve tilts — that decision
belongs to a separate committee that reads your summary next.
````

- [ ] **Step 2: Write `prompts/02_options_analysis.md`**

````markdown
# Stage 2 — Committee Decision on Today's Sizing Options

You are the investment-committee chair at a macro asset-management desk — the
qualitative barrier between the news and the orders. Today is {DATE}.

The systematic engine produces one sizing; two deterministic tilts of it are also
on the table. You must rank the three and recommend exactly one. You cannot
invent weights: the only weights that may be traded are the three option tables
below.

## Executive summary from the macro strategist (Stage 1)

{STAGE1_TEXT}

## The three options (exact weights; "note" flags any deterministic repair)

```json
{OPTIONS_JSON}
```

## How to decide

1. Weigh the strategist's regime evidence against what each option actually
   holds (equity-momentum A+B, rates, cash ballast, CTA) and its turnover cost.
2. Consider the **veto list** — veto forces the `systematic` option:
   - Stage-1 summary unavailable or self-reported low confidence on a major
     market-moving day
   - stale/missing metrics the decision depended on ("n/a" cells)
   - contradictory macro shocks the systematic engine cannot see
     (overnight geopolitical shock, unscheduled central-bank action)
3. Base rates matter: `systematic` is the researched, non-overfit baseline.
   Depart from it only when the news evidence is clear and current.

## Output format — follow exactly

## Assessment
At most 3 short paragraphs: the regime call, what each option would do about it,
and whether any veto condition fired (say which).

## Option ranking
1. <systematic|risk_on|risk_off> — one-line reason
2. <systematic|risk_on|risk_off> — one-line reason
3. <systematic|risk_on|risk_off> — one-line reason

End your response with exactly these three lines and nothing after them:

RECOMMENDED_OPTION: <systematic|risk_on|risk_off>
CONFIDENCE: <low|med|high>
VETO: <yes|no>
````

- [ ] **Step 3: Verify tokens and contract lines**

Run: `python -c "from live.discretionary import _render_prompt; t=_render_prompt('01_news_exec_summary.md', DATE='2026-08-27', METRICS_JSON='{}'); assert '{DATE}' not in t and '{METRICS_JSON}' not in t; t=_render_prompt('02_options_analysis.md', DATE='2026-08-27', STAGE1_TEXT='x', OPTIONS_JSON='{}'); assert '{STAGE1_TEXT}' not in t and '{OPTIONS_JSON}' not in t; print('prompt tokens OK')"`
Expected: `prompt tokens OK`

- [ ] **Step 4: Commit**

```bash
git add prompts/01_news_exec_summary.md prompts/02_options_analysis.md
git commit -m "docs(prompts): add stage-1 exec-summary and stage-2 committee SOPs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Wire `morning_report.py` + extend the PDF

**Files:**
- Modify: `scripts/morning_report.py` (rewire main; new record schema)
- Modify: `live/discretionary.py` (`build_report` new signature/sections; delete old `analyze()` + `_parse_analysis`)

- [ ] **Step 1: Rewrite `scripts/morning_report.py`** — full new file content (replaces everything, including the now-deleted `_no_analysis`; `_parse_args` gains `--no-llm` coverage of both stages):

```python
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
        prices, sys_targets["ticker_weights"], latest_a, latest_b,
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
```

- [ ] **Step 2: Replace `build_report` and delete the old single-stage API in `live/discretionary.py`**

Delete `analyze()` and `_parse_analysis()` entirely. Replace `build_report` with:

```python
def _kv_table(d: Dict) -> str:
    rows = [f"{_tex_escape(k)} & {_tex_escape(v)} \\\\" for k, v in d.items()]
    return ("\\begin{tabular}{ll}\n\\hline\nMetric & Value \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}")


def _ticker_delta_table(tickers: Dict[str, float], base: Dict[str, float],
                        equity: float) -> str:
    rows = []
    for t, w in sorted(tickers.items(), key=lambda kv: kv[1], reverse=True):
        d = 100.0 * (w - base.get(t, 0.0))
        rows.append(f"{_tex_escape(t)} & {100*w:.1f} & {d:+.1f} & {_dollars(w, equity)} \\\\")
    return ("\\begin{tabular}{lrrr}\n\\hline\n"
            "Ticker & Weight (\\%) & $\\Delta$ pp & \\$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}")


def build_report(
    run_date: date,
    systematic: Dict,
    metrics: Dict,
    options: Dict[str, Dict],
    stage1: Dict,
    stage2: Dict,
    equity: float,
    out_pdf: Path,
) -> Path:
    """Render the LaTeX report to ``out_pdf`` via pdflatex. Returns the PDF path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    build_dir = LOG_DIR / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    sections.append(
        "\\section*{Original systematic sizings}\n"
        f"Account equity: \\${equity:,.0f}\n\n"
        f"\\subsection*{{Sleeve}}\n{_sleeve_table(systematic['sleeve'], equity)}\n\n"
        f"\\subsection*{{Tickers}}\n{_ticker_table(systematic['tickers'], equity)}")

    panel = {k: metrics.get(k) for k in (
        "vix_close", "vix_change_1d", "vix_percentile_252d", "vix_overlay_active",
        "tnx_10y_level", "tnx_change_5d", "tlt_above_sma200", "ief_above_sma200",
        "breadth_pct_above_sma200", "holdings_below_sma200", "equity",
        "peak_equity", "drawdown_pct", "guardrail_margin_pct", "turnover_oneway_pct")}
    sections.append("\\section*{Metrics panel}\n" + _kv_table(panel))

    s1 = (stage1.get("exec_summary") or "(unavailable)")
    regime = stage1.get("regime_bias") or "n/a"
    sections.append(
        "\\section*{Executive summary}\n"
        + _tex_escape(s1).replace("\n", "\n\n")
        + f"\n\nRegime bias: \\textbf{{{_tex_escape(regime)}}} "
          f"(confidence: {_tex_escape(stage1.get('summary_confidence') or 'n/a')})")

    base = systematic["tickers"]
    for name in OPTION_NAMES:
        o = options[name]
        note = f"\\\\\nNote: {_tex_escape(o['note'])}" if o.get("note") else ""
        sections.append(
            f"\\section*{{Option: {name}}}\n"
            f"\\subsection*{{Sleeve}}\n{_sleeve_table(o['sleeve'], equity)}\n\n"
            f"\\subsection*{{Tickers (delta vs systematic)}}\n"
            f"{_ticker_delta_table(o['tickers'], base, equity)}{note}")

    ranking = "\n\n".join(
        f"{r['rank']}. \\textbf{{{_tex_escape(r['option'])}}} --- {_tex_escape(r['reason'])}"
        for r in stage2.get("ranking", []) if r.get("option"))
    rec = stage2.get("recommended_option") or "none"
    if stage2.get("veto") == "yes":
        rec = "systematic (VETO)"
    sections.append(
        "\\section*{Committee decision}\n"
        + _tex_escape(stage2.get("assessment") or "(unavailable)").replace("\n", "\n\n")
        + (f"\n\n{ranking}" if ranking else "")
        + f"\n\nRecommended option: \\textbf{{{_tex_escape(rec)}}} "
          f"(confidence: {_tex_escape(stage2.get('confidence') or 'n/a')}, "
          f"veto: {_tex_escape(stage2.get('veto') or 'n/a')})")

    sections.append(
        "\\section*{How to execute}\nPick one option and run:\\\\\n"
        f"\\texttt{{python scripts/rebalance.py --option "
        f"{stage2.get('recommended_option') or '<name>'} --date {run_date.isoformat()}}}")

    doc = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[a4paper,margin=2cm]{geometry}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\title{Discretionary morning report\\\\\\large " + run_date.isoformat() + "}\n"
        "\\date{}\n\\begin{document}\n\\maketitle\n"
        + "\n\n".join(sections) + "\n\\end{document}\n"
    )

    tex_path = build_dir / f"morning_report_{run_date.isoformat()}.tex"
    tex_path.write_text(doc, encoding="utf-8")

    pdflatex = _find_pdflatex()
    if pdflatex:
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={build_dir}", str(tex_path)],
            capture_output=True, text=True, timeout=120)
        built = build_dir / tex_path.with_suffix(".pdf").name
        if built.exists():
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built), str(out_pdf))
            return out_pdf
    print(f"WARN: pdflatex not found or failed; .tex left at {tex_path}")
    return tex_path
```

Also update the module docstring first paragraph to: "Two-stage discretionary 07:30 macro-news overlay for the A+B+Diversifier Sleeves strategy. The LLM never touches weights: Stage 1 emits an executive summary, Stage 2 a committee ranking; all sizings come from deterministic tilt rules. See prompts/01_news_exec_summary.md and prompts/02_options_analysis.md."

- [ ] **Step 3: Rewrite the module self-check** — replace `_self_check()` in `live/discretionary.py`:

```python
def _self_check() -> None:
    # Tilt engine on a neutral toy sleeve (sums to 1, caps, ordering).
    sleeve = pd.Series({"A": 0.20, "B": 0.20, "rates": 0.20,
                        "BIL_ballast": 0.20, "cta": 0.20})
    on, _ = _tilt_risk_on(sleeve)
    off, _ = _tilt_risk_off(sleeve, rates_in_uptrend=True)
    for v in (on, off):
        assert abs(v.sum() - 1.0) < 1e-9
    assert abs(on["A"] + on["B"] - 0.60) < 1e-9
    assert abs(off["A"] + off["B"] - 0.20) < 1e-9
    assert abs(off["rates"] - 0.45) < 1e-9  # cap binds; remainder spills to BIL_ballast
    on2, note2 = _tilt_risk_on(pd.Series({"A": 0.3, "B": 0.3, "rates": 0.05,
                                          "BIL_ballast": 0.05, "cta": 0.30}))
    assert note2 == "risk-on tilt unfundable; systematic kept"
    # Parsers accept valid control lines and reject bad enums.
    s1 = _parse_stage1("## Today's events\n- 08:30 ET: CPI (BLS)\n\nx\n"
                       "REGIME_BIAS: risk_off\nSUMMARY_CONFIDENCE: high\n")
    assert s1["regime_bias"] == "risk_off" and s1["macro_calendar"] == ["08:30 ET: CPI (BLS)"]
    s2 = _parse_stage2("## Option ranking\n1. risk_on — breadth strong.\n\n"
                       "RECOMMENDED_OPTION: risk_on\nCONFIDENCE: med\nVETO: no\n")
    assert s2["recommended_option"] == "risk_on" and s2["ranking"][0]["option"] == "risk_on"
    assert _parse_stage2("RECOMMENDED_OPTION: yolo\nVETO: maybe\n")["veto"] == ""
    # Prompt files render with all tokens substituted.
    t = _render_prompt("01_news_exec_summary.md", DATE="2026-08-27", METRICS_JSON="{}")
    assert "{DATE}" not in t and "{METRICS_JSON}" not in t
    print("discretionary self-check OK: tilt engine, parsers, prompt tokens")


if __name__ == "__main__":
    _self_check()
```

- [ ] **Step 4: Run the deterministic report end-to-end**

Run: `python -m live.discretionary`
Expected: `discretionary self-check OK: tilt engine, parsers, prompt tokens`

Run: `python scripts/morning_report.py --date 2025-06-30 --no-llm --prefer-yfinance`
Expected: exit 0; prints `Wrote ...logs\discretionary_2025-06-30.json` and a PDF (or `WARN: pdflatex ...` leaving a `.tex`); final lines say `Committee recommendation: none` and show the `--option` command.

Then inspect the JSON: `python -c "import json; d=json.load(open('logs/discretionary_2025-06-30.json')); print(sorted(d)); print(sorted(d['options'])); print(sorted(d['metrics']))"`
Expected: `['date', 'equity', 'metrics', 'options', 'stage1', 'stage2', 'systematic']`, `['risk_off', 'risk_on', 'systematic']`, and the metrics keys.

- [ ] **Step 5: Run the full test suite green**

Run: `python -m pytest tests/ -q`
Expected: all PASS (existing tests + the three added).

- [ ] **Step 6: Commit**

```bash
git add scripts/morning_report.py live/discretionary.py
git commit -m "feat(report): wire two-stage overlay into morning report; metrics + options + committee sections in PDF

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: `rebalance.py` `--option` + remove the fixed profiles

**Files:**
- Modify: `scripts/rebalance.py`
- Modify: `live/discretionary.py` (delete `PROFILES` + `apply_profile` — superseded by `build_tilt_options`)

- [ ] **Step 1: Update `scripts/rebalance.py`**

In the imports replace `from live.discretionary import PROFILES, apply_profile` with:

```python
from live.discretionary import OPTION_NAMES
```

In `_parse_args` replace the `--profile` argument with:

```python
    p.add_argument("--option", choices=list(OPTION_NAMES), default=None,
                   help="Trade a discretionary option from logs/discretionary_<date>.json")
```

Update the module docstring lines 13-16 to: "With ``--option {systematic,risk_on,risk_off}`` it instead trades the discretionary
option chosen from the 07:30 morning report (``logs/discretionary_<date>.json``);
weights come from the report, not from the systematic engine."
Update the usage line 26 to: `  python scripts/rebalance.py --option risk_off [--date YYYY-MM-DD]`

Replace `_load_profile_tickers` with:

```python
def _load_option(run_date: date, option: str) -> Tuple[Dict[str, float], pd.Series]:
    """Read the chosen option's ticker + sleeve weights from the morning report JSON."""
    path = LOG_DIR / f"discretionary_{run_date.isoformat()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"discretionary report not found: {path}. Run scripts/morning_report.py "
            f"--date {run_date.isoformat()} first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    o = data["options"][option]
    return o["tickers"], pd.Series(o["sleeve"], dtype=float)
```

In `main()` replace the `elif args.profile:` branch with:

```python
    elif args.option:
        try:
            target_tickers, sleeve_weights = _load_option(run_date, args.option)
        except Exception as e:
            print(f"FATAL: could not load option '{args.option}': {e}")
            return 1
        print(f"Using discretionary option '{args.option}' "
              f"({len(target_tickers)} tickers).")
```

And update the startup print: `profile={args.profile}` → `option={args.option}`.

- [ ] **Step 2: Delete `PROFILES` and `apply_profile` from `live/discretionary.py`** (both are superseded by `build_tilt_options`; nothing imports them after Step 1). Also delete the `_SLEEVE_ORDER = SLEEVE_ORDER` alias line and update `_sleeve_table` to iterate `SLEEVE_ORDER`.

Also add veto enforcement inside `_load_option`, just before the `return`:

```python
    if data.get("stage2", {}).get("veto") == "yes" and option != "systematic":
        print(f"VETO ACTIVE in report: overriding --option {option} -> systematic")
        o = data["options"]["systematic"]
```



- [ ] **Step 3: Verify the option path trades**

Run: `python scripts/rebalance.py --option risk_off --date 2025-06-30 --dry-run --prefer-yfinance`
(Uses the JSON written in Task 5 Step 4. Dry-run: no Alpaca calls.)
Expected: exit 0, prints `Using discretionary option 'risk_off' ...`, sleeve/ticker targets, and either orders or the drift-skip line; no `FATAL`.

Run: `python scripts/rebalance.py --profile balanced --date 2025-06-30 --dry-run`
Expected: argparse error `invalid choice` / `unrecognized arguments` (exit 2) — the old flag is gone.

- [ ] **Step 4: Run module self-check + full tests**

Run: `python -m live.discretionary` then `python -m pytest tests/ -q`
Expected: `discretionary self-check OK...` and all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/rebalance.py live/discretionary.py
git commit -m "feat(live): replace fixed --profile with --option trading the tilted sizings

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: README §8b rewrite

**Files:**
- Modify: `README.md` (section `## 8b. Discretionary 07:30 report (news overlay)`)

- [ ] **Step 1: Replace section 8b with:**

````markdown
## 8b. Discretionary 07:30 report (news overlay v2)

An optional two-stage discretionary macro-news overlay that **does not change the
reproducible/anti-overfit strategy**. The LLM only advises; all weights are
deterministic.

Flow:

1. **07:30 (scheduled)** — `python scripts/morning_report.py` computes today's
   systematic targets and a **metrics panel** (VIX level + 252d percentile and
   overlay status, 10Y yield + 5d change, TLT/IEF trend, breadth above SMA200,
   holdings below SMA200, drawdown vs the -10% guardrail, implied turnover),
   runs **Stage 1** (headless `claude` with web search, SOP in
   `prompts/01_news_exec_summary.md`) for an executive summary + regime bias,
   builds three **deterministic tilt options of today's sizings**
   (`systematic` as-is; `risk_on` = A/B/CTA x1.5 funded from BIL ballast then
   rates; `risk_off` = A/B/CTA x0.5 with proceeds to rates when bonds are in
   uptrend, else BIL; caps: 45% per risk sleeve, 70% A+B, 35% per ticker;
   risk_on is disabled when the VIX overlay is active), then runs **Stage 2**
   (SOP in `prompts/02_options_analysis.md`) for a committee ranking,
   recommendation, confidence and veto flag. Writes:
   - `logs/morning_report_<date>.pdf` — metrics panel, exec summary, the three
     option tables with deltas vs systematic, and the committee decision.
   - `logs/discretionary_<date>.json` — full record (metrics, options, stages).
2. **You read the PDF and pick one option.**
3. **You trade it** — `python scripts/rebalance.py --option <systematic|risk_on|risk_off>`
   loads that option's weights from the JSON and sends balancing orders (paper by
   default). Risk guardrails still apply. `VETO: yes` in the report means trade
   `systematic` regardless of the recommendation.

The LLM never edits weights — it emits prose and control-line labels only; both
stages degrade gracefully to the deterministic tables if the CLI call fails, and
the SOPs can be retuned by editing the Markdown in `prompts/` (no code changes).

Self-check: `python -m live.discretionary` and `python -m live.morning_metrics`.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite 8b for two-stage news overlay with tilted options

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Final verification

- [ ] **Step 1: Full suite + self-checks**

Run: `python -m pytest tests/ -q && python -m live.discretionary && python -m live.morning_metrics`
Expected: all PASS; both self-checks print OK lines.

- [ ] **Step 2: Live two-stage dry run (manual, costs two claude calls)**

Run: `python scripts/morning_report.py --prefer-yfinance`
Expected: exit 0; Stage-1 regime bias and committee recommendation printed; PDF written without pdflatex warnings; JSON `stage1.source` and `stage2.source` = `claude-cli`. Read the PDF once for sanity (metrics table populated, deltas shown, ranking present).

- [ ] **Step 3: Confirm scheduler needs no change**

Run: `cat _run_rebalance_0731.bat`
Expected: it still calls `scripts/rebalance.py` without `--profile`/`--option` (the bat runs the systematic path; the picked option is traded manually). Only edit if it references the removed `--profile` flag.
