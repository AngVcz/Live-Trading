# News Overlay v2 — Metrics Panel, Two-Stage LLM, Tilted Options

**Date:** 2026-08-26
**Status:** Approved design, pre-implementation
**Repo:** `Live/` (this file lives in the Live repo; the system it describes is the Live trading stack)

---

## 1. Context

`Live/scripts/morning_report.py` already runs the discretionary 07:30 flow: compute
systematic targets → single headless `claude` web-search analysis → recommend one of
three **fixed** sleeve profiles (aggressive/balanced/passive) → LaTeX PDF +
`logs/discretionary_<date>.json` → human picks → `rebalance.py --profile`.

The user (acting as the macro asset manager's qualitative barrier) wants the upgraded
flow: **News Scraping → Executive Summary with key resizing metrics → 3 options built
by tilting today's actual algorithmic sizings**, expressed as two `.md` SOP documents
that drive two independent LLM stages.

Carried-over hard rule: **the LLM never emits weights.** It produces prose plus
control-line labels only; every weight in the report comes from deterministic Python.

## 2. Decision

Approach 1 (approved 2026-08-26): two-stage prompt-pack + deterministic tilt engine,
built by **extending** the existing morning report in place. The fixed `PROFILES`
vectors and `apply_profile()` in `live/discretionary.py` are **deleted** and replaced
by the tilt engine (§5). `--profile` is replaced by `--option` (§7).

## 3. Pipeline

```
07:30 scripts/morning_report.py                      (entry point unchanged)
 1. compute_systematic_targets()                     [existing, rebalance.py]
 2. compute_metrics_panel(...) -> dict               [NEW live/morning_metrics.py]
 3. STAGE 1 LLM: prompts/01_news_exec_summary.md
      rendered with {date, metrics_json, sleeve_context}
      -> exec summary prose + REGIME_BIAS/SUMMARY_CONFIDENCE control lines
 4. build_tilt_options(systematic_sleeve)            [NEW, live/discretionary.py]
      -> 3 deterministic options (systematic / risk_on / risk_off),
         each decomposed to tickers via decompose_target_to_tickers()
 5. STAGE 2 LLM: prompts/02_options_analysis.md
      rendered with {stage1_text, option_tables}
      -> assessment + ranking + RECOMMENDED_OPTION/CONFIDENCE/VETO
 6. PDF (extended section order) + logs/discretionary_<date>.json (new schema, §8)

user picks -> scripts/rebalance.py --option <systematic|risk_on|risk_off>
```

Two LLM calls via the existing headless-`claude` mechanism (reuse the
`subprocess` invocation in `discretionary.analyze`, factored into a shared
`_call_claude(prompt, model)` helper; 300 s timeout each, `--allowedTools WebSearch`,
`--output-format json`). No `anthropic` SDK, no new secrets, no new Python
dependencies.

## 4. Metrics panel — `live/morning_metrics.py`

```python
def compute_metrics_panel(
    prices: pd.DataFrame,          # panel from compute_systematic_targets
    ticker_weights: dict,          # today's systematic ticker targets
    weights_a: pd.Series,          # weights_a_last
    weights_b: pd.Series,          # weights_b_last
    equity: float,
    as_of: date,
) -> dict:
```

All values computed in Python; the LLM receives them as authoritative JSON and may
**interpret** but not restate or extend them. On any single metric failing (missing
column, short history, fetch error) the cell is `"n/a"` and the report continues.

| Key(s) | Computation |
|---|---|
| `vix_close`, `vix_change_1d` | last two closes of `^VIX` column |
| `vix_percentile_252d` | `scipy.stats.percentileofscore` of last VIX close vs trailing 252 obs, min 126 — mirrors `core_signals` |
| `vix_overlay_active` | `vix_percentile_252d > 0.70` (the existing ½-cut rule) |
| `tnx_10y_level`, `tnx_change_5d` | one extra `fetch_panel(["^TNX"])` call (cached by data_feed); `n/a` on failure |
| `tlt_above_sma200`, `ief_above_sma200` | booleans; defines whether the rates sleeve is in a bond uptrend |
| `breadth_pct_above_sma200` | % of `core_signals.UNIVERSE` tickers present in the panel with close > SMA200 |
| `holdings_below_sma200` | list of current A top-5 / B top-3 holdings below their SMA200 (at-risk names) |
| `equity`, `peak_equity`, `drawdown_pct` | equity + `state.load_state()` peak |
| `guardrail_margin_pct` | distance from current equity to the −10% drawdown guardrail |
| `turnover_oneway_pct` | Σ\|w_today − w_last\| / 2, using `portfolio.load_last_weights()` |
| `sleeve_weights`, `top_tickers` | today's sleeve vector + top-10 ticker weights (echoed for the prompt) |
| `macro_calendar` | empty placeholder; Stage 1 fills it from web search |

## 5. Tilt engine — `build_tilt_options()` in `live/discretionary.py`

Input: systematic sleeve Series `s` over `[A, B, rates, BIL_ballast, cta]` (plus
`latest_a`, `latest_b`, `prices` for ticker decomposition). Output: dict of three
options, each `{"sleeve": {...}, "tickers": {...}}`.

- **`systematic`** — `s` unchanged. Baseline; always present, always executable.
- **`risk_on`** — `A×1.5, B×1.5, cta×1.5`. The increase (≤ +0.30 NAV given systematic
  bounds) is funded from `BIL_ballast` first, then `rates`; neither may go negative.
- **`risk_off`** — `A×0.5, B×0.5, cta×0.5`. Proceeds go to `rates` if the rates pick
  is in an uptrend (`tlt_above_sma200 or ief_above_sma200` from the metrics panel,
  passing zeros if `n/a`), else to `BIL_ballast`.

Post-tilt bounds, always enforced (repair by scaling tilted sleeves back toward `s`,
then renormalize to sum 1.0):

1. each sleeve ≤ 0.45
2. `A + B` ≤ 0.70
3. after ticker decomposition: any single ticker > 0.35 is clipped, spill to `BIL`
   (leaves margin below the 0.50 NAV guardrail in `risk.py`)
4. if the metrics panel shows `vix_overlay_active`, the systematic engine has
   already halved A/B: `risk_on` is then **identical to `systematic`** (tilt
   disabled) and its `note` field carries "risk-on tilt disabled by active VIX
   overlay". No estimate of any "pre-cut" share is made.

Output option dicts are `{"sleeve": {...}, "tickers": {...}, "note": ""}` (`note` is
empty unless a bound repair or the overlay rule fired).

Ordering invariant asserted in self-check: `(A+B)_risk_off ≤ (A+B)_systematic ≤ (A+B)_risk_on`.

## 6. Prompt SOPs — `Live/prompts/`

Two Markdown files, read at runtime and injected verbatim into the two `claude` calls.
Tuning the analysts = editing these files; no code changes. Both files end with the
output contract and the sentence: "The metrics JSON is authoritative. Cite a source
name for every news claim. You may not propose portfolio weights."

### `01_news_exec_summary.md` — role: macro strategist

- Injected: run date, metrics panel JSON, one-paragraph sleeve description.
- Web-search brief: Fed/ECB, CPI/PCE, employment, GDP, geopolitics, Treasury yields,
  VIX, commodities, news on current holdings, and **today's macro calendar**.
- Output contract (parsed):
  - `## Headlines` — ≤5 bullets (`- ` prefix)
  - `## Macro regime` — evidence table, prose
  - `## Today's events` — macro calendar lines (populates `macro_calendar`)
  - `## Metrics read` — interpretation of the panel; no new numbers
  - final two lines: `REGIME_BIAS: <risk_on|neutral|risk_off>` then
    `SUMMARY_CONFIDENCE: <low|med|high>`
- Forbidden: any portfolio sizing advice (that is Stage 2's job).

### `02_options_analysis.md` — role: investment-committee chair (the qualitative barrier)

- Injected: Stage-1 summary text + the three option tables (sleeve + ticker weights
  and Δ vs systematic).
- Task: weigh regime evidence against each option's exposures; rank; explicitly check
  the veto list (news blackout/stage-1 unavailable, stale data flagged in metrics,
  contradictory macro shocks the systematic engine cannot see).
- Output contract (parsed):
  - `## Assessment` — ≤3 short paragraphs
  - `## Option ranking` — three lines `1. <option> — <reason>`
  - final three lines: `RECOMMENDED_OPTION: <systematic|risk_on|risk_off>`,
    `CONFIDENCE: <low|med|high>`, `VETO: <yes|no>`
- `VETO: yes` ⇒ executor trades `systematic` regardless of recommendation.

Enum note (kept deliberately distinct): Stage 1 speaks **regime** language
(`risk_on|neutral|risk_off`); Stage 2 speaks **option** language
(`systematic|risk_on|risk_off`). A `neutral` regime is an input to Stage 2's
judgment, not a mechanical mapping to `systematic`.

### Parsers

`discretionary.py` gains `_parse_stage1()` / `_parse_stage2()` mirroring the existing
regex style: strict enum validation, unknown labels → `""`, control lines stripped
from prose. Missing/garbled stage output never raises into the report build.

## 7. Execution path

- `rebalance.py`: `--profile {aggressive,balanced,passive}` is **replaced** by
  `--option {systematic,risk_on,risk_off}`, reading
  `options[<name>]["tickers"]` from today's JSON. `--weights` JSON override stays.
  Risk guardrails apply unchanged (they run downstream of weight selection).
- Historical `discretionary_*.json` files with the old `profiles` schema remain
  readable as artifacts but are no longer tradeable. Acceptable: none should be
  re-traded after the fact.
- `README.md` §8b rewritten for the new flow; the fixed-profiles table is replaced
  by the tilt-rules description.
- Scheduler unchanged: 07:30 report task; rebalance stays a human-picked step.
  If the two LLM stages push runtime past the pick-then-trade window, move the
  scheduled task earlier (operational tweak, not code).
- `_run_rebalance_0731.bat` untouched.

## 8. JSON schema — `logs/discretionary_<date>.json`

```json
{
  "date": "2026-08-26",
  "equity": 100000.0,
  "systematic": {"sleeve": {...}, "tickers": {...}},
  "metrics": { "vix_close": ..., "vix_percentile_252d": ..., "vix_overlay_active": true,
               "tnx_10y_level": ..., "tnx_change_5d": ..., "tlt_above_sma200": true,
               "ief_above_sma200": false, "breadth_pct_above_sma200": ...,
               "holdings_below_sma200": [...], "equity": ..., "peak_equity": ...,
               "drawdown_pct": ..., "guardrail_margin_pct": ...,
               "turnover_oneway_pct": ..., "sleeve_weights": {...},
               "top_tickers": {...}, "macro_calendar": [] },
  "options": { "systematic": {"sleeve": {...}, "tickers": {...}, "note": ""},
               "risk_on":    {"sleeve": {...}, "tickers": {...}, "note": ""},
               "risk_off":   {"sleeve": {...}, "tickers": {...}, "note": ""} },
  "stage1": { "exec_summary": "...", "headlines": [...], "macro_calendar": [...],
              "regime_bias": "neutral", "summary_confidence": "med",
              "source": "claude-cli|none" },
  "stage2": { "assessment": "...", "ranking": [{"rank": 1, "option": "risk_on", "reason": "..."}],
              "recommended_option": "risk_on", "confidence": "med",
              "veto": "no", "source": "claude-cli|none" }
}
```

## 9. Report (PDF)

Section order: (1) Original systematic sizings, (2) **Metrics panel** (key/value
table), (3) **Executive summary** (Stage-1 prose + headlines + regime bias),
(4) Options — three tables, each weight shown with Δ vs systematic, (5) **Committee
decision** (ranking, recommendation, confidence, veto), (6) How to execute
(`rebalance.py --option <name>`). LaTeX rendering stays in `build_report()`; missing
pdflatex keeps the existing ".tex left behind" fallback.

## 10. Failure modes

| Failure | Behavior |
|---|---|
| Stage-1 LLM fails | Report ships with "analysis unavailable"; `stage1.source = "none"`; options tables still valid |
| Stage-2 LLM fails | No recommendation; `veto` empty; human decides unaided |
| Single metric fails | `"n/a"` cell; report continues |
| `^TNX` fetch fails | rates metrics `n/a` |
| pdflatex missing | `.tex` left in `logs/_build` (existing) |
| Any LLM attempt to emit weights | parsers ignore everything except control-line enums |

## 11. Testing

- `python -m live.discretionary` — extended self-check: three options sum to 1.0,
  caps hold, ordering invariant holds, stage-1/stage-2 parsers accept valid samples
  and reject bad enum values, `risk_on ≡ systematic` when the VIX overlay is active.
- `python -m live.morning_metrics` — self-check on cached panel: required keys
  present, percentile in [0,1], booleans are bools.
- `tests/test_live_pipeline.py` — one added smoke test: `build_tilt_options` on a
  toy sleeve vector stays within bounds and decomposes. No Alpaca calls (existing
  convention).

## 12. Files touched

| File | Change |
|---|---|
| `Live/scripts/morning_report.py` | wire metrics + two stages + new JSON schema |
| `Live/live/discretionary.py` | tilt engine, stage parsers, `_call_claude`, PDF sections; **delete** `PROFILES`/`apply_profile` |
| `Live/live/morning_metrics.py` | **new** — `compute_metrics_panel` + self-check |
| `Live/prompts/01_news_exec_summary.md` | **new** — Stage-1 SOP |
| `Live/prompts/02_options_analysis.md` | **new** — Stage-2 SOP |
| `Live/scripts/rebalance.py` | `--profile` → `--option` |
| `Live/tests/test_live_pipeline.py` | one smoke test |
| `Live/README.md` | §8b rewrite |

## 13. Non-goals

- No Python-side news scraping (RSS/API clients): web search happens inside the
  LLM stages via `--allowedTools WebSearch`.
- No dynamic/LLM-generated weights, ever.
- No new Python dependencies (scipy + MiKTeX already present).
- No changes to the systematic engine, guardrails, Alpaca executor, or scheduler.
- No backtest of the overlay itself (it is a human-in-the-loop process, not a
  tradeable signal).
