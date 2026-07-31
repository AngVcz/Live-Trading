# A+B+Diversifier Sleeves — Live Trading Implementation

This folder (`Live/`) contains everything needed to run the **A+B+Diversifier Sleeves** strategy live with Alpaca. Nothing else is used for implementation; all production code lives here.

---

## 1. Strategy

### Idea

Combine two equity dual-momentum engines (A and B) with three truly diversifying sleeves to build a robust, non-overfit, live-tradeable portfolio.

### Target allocation (sleeves)

| Sleeve        | Weight | Description |
|---------------|--------|-------------|
| Strategy A    | 20%    | Phase 3 mom_corr: momentum + correlation, top 5 |
| Strategy B    | 20%    | Top-3 Dual-Momentum: pure momentum, top 3 |
| Rates sleeve  | 20%    | Bond trend: TLT / IEF / BIL |
| Bear sleeve   | 20%    | SH when SPY < SMA200, otherwise BIL |
| CTA proxy     | 20%    | PDBC / DBMF / KMLM in uptrend, otherwise BIL |

### Why this structure

- A and B are highly correlated, so switching rules between them add no OOS value.
- C (Triple EMA + Macro + Kurt) was discarded as a primary engine due to look-ahead and overfit risk.
- The three sleeves (Rates, Bear, CTA proxy) break correlation and approximately double Sharpe and triple Calmar in backtest.
- Gold is omitted because it degrades Calmar in the OOS sample.

---

## 2. Entry / Exit

### Core engines (A and B)

- **Momentum signal**: RSI(14) on adaptive cumulative return over 63/126/252 days, blended by VIX percentile.
- **Slow filter**: price above 252-day SMA.
- **Fast filter**: EMA 8/21/50 + MACD 12/26/9 aligned bullish.
- **Selection**: every Friday, pick the top-N assets by composite score.
- **Risk overlay**: if VIX is above its 70th percentile (last 252 days), cut exposure by half.

### Portfolio rebalancing

- **Base frequency**: annual (first 5 trading days of January).
- **Drift threshold**: if any sleeve deviates more than ±10% from its target weight, rebalance.
- **Execution**: sells first, then buys, to free buying power.

### Costs

- 10 bps one-way in backtest (adjust with observed real slippage in paper trading).
- Alpaca ETFs are commission-free, but SH/PSQ/VIXY may have wider spreads.

---

## 3. Sizing

### Sleeve level

```
A       = 20%
B       = 20%
rates   = 20%
bear    = 20%
cta     = 20%
```

### Ticker level

- A and B: equal-weight within their selected top-N.
- Rates: 100% in the best of TLT / IEF / BIL based on SMA200.
- Bear: 100% in SH if SPY < SMA200, otherwise BIL.
- CTA proxy: equal-weight among PDBC / DBMF / KMLM in uptrend, otherwise BIL.

### Account scaling

- The runner reads Alpaca account equity and allocates dollars proportionally.
- Fractional shares are supported by default; if unavailable, it rounds to whole lots.

---

## 4. Ensemble

The strategy is a **fixed-weight** ensemble at the sleeve level, not dynamic. Research showed that A/B/C switching rules added overfit and failed to beat A or B alone OOS.

- A and B are two implementations of the same dual-momentum engine with different scoring and concentration.
- Each contributes 20% of NAV.
- The diversifier sleeves contribute 60% of NAV.

---

## 5. File structure

```
Live/
├── README.md
├── pyproject.toml             # Project metadata + deps + pytest config
├── requirements.txt            # `pip install -r requirements.txt`
├── .env.example                # Template; copy to .env (gitignored)
├── live/                       # Strategy package
│   ├── __init__.py
│   ├── core_signals.py         # Point-in-time engines A and B
│   ├── data_feed.py            # Alpaca + yfinance + cache
│   ├── alpaca_executor.py      # Order placement
│   ├── portfolio.py            # Sleeve construction and targets
│   ├── risk.py                 # Safety guardrails
│   ├── state.py                # Peak equity / weights persistence
│   └── monitor.py              # Live vs backtest monitoring
├── scripts/
│   └── rebalance.py            # Daily entry point (was live_runner.py)
├── notebooks/
│   └── A_B_Diversifier_Analytics.ipynb
├── tests/
│   ├── __init__.py
│   └── test_live_pipeline.py   # Smoke tests
└── logs/                       # Runtime artifacts (gitignored): state.json, orders_*.csv, target_weights.jsonl
    └── .gitkeep
```

---

## 6. Setup

### Installation

```bash
cd Live
pip install -r requirements.txt
```

### Credentials

Create a `.env` file inside `Live/` with:

```env
ALPACA_API_KEY=PK...
ALPACA_API_SECRET=...
ALPACA_LIVE=false
```

**Never commit `.env` to Git.** It is already in `.gitignore`.

---

## 7. Usage

### Dry-run (does not touch Alpaca)

```bash
cd Live
python scripts/rebalance.py --date 2025-06-27 --dry-run --prefer-yfinance
```

This uses yfinance, simulates a $100,000 account, and prints the orders it would send.

### Paper trading

1. Make sure `ALPACA_LIVE=false`.
2. Remove `--dry-run`:

```bash
cd Live
python scripts/rebalance.py --prefer-yfinance
```

### Real live trading

1. Validate several months in paper.
2. Set `ALPACA_LIVE=true`.
3. Run without `--dry-run`.

---

## 8. Scheduler

Run once per day after market close. Example cron on Linux/macOS:

```cron
35 16 * * 1-5 cd /path/to/Live && python scripts/rebalance.py
```

On Windows use Task Scheduler or Git Bash cron.

---

## 8b. Discretionary 07:30 report (news overlay)

An optional discretionary macro-news overlay that **does not change the
reproducible/anti-overfit strategy**. The LLM only advises; all weights are
deterministic.

Flow:

1. **07:30 (scheduled)** — `python scripts/morning_report.py` computes today's
   systematic targets, asks the LLM (headless `claude` CLI with web search) for a
   market analysis + a recommended profile, builds three deterministic sizing
   combinations, and writes:
   - `logs/morning_report_<date>.pdf` — the LaTeX report (original sizings, market
     analysis, recommended profile, and three combinations: aggressive / balanced /
     passive).
   - `logs/discretionary_<date>.json` — the three profiles' sleeve + ticker weights.
2. **You read the PDF and pick one profile.**
3. **You trade it** — `python scripts/rebalance.py --profile <aggressive|balanced|passive>`
   loads that profile's weights from the JSON and sends balancing orders (paper by
   default). Risk guardrails still apply.

The three profiles are fixed regime-gate sleeve allocations (each sums to 100%):

| Profile    | A    | B    | rates | bear | cta  | tilt     |
|------------|------|------|-------|------|------|----------|
| aggressive | 30%  | 25%  | 10%   | 5%   | 30%  | risk-on  |
| balanced   | 20%  | 20%  | 20%   | 20%  | 20%  | neutral  |
| passive    | 10%  | 10%  | 30%   | 25%  | 25%  | risk-off |

The LLM picks one of these; it never edits the numbers. If the LLM call fails, the
report still ships the three deterministic tables with an "analysis unavailable"
note. No `anthropic` SDK or extra API key is required — the headless CLI reuses
your existing Claude Code auth.

Self-check: `python -m live.discretionary`.

## 9. Risk guardrails

The runner automatically blocks execution if:

- Price data is more than 2 days stale.
- Any target ticker is missing from the price panel.
- A single position exceeds 50% of NAV.
- A+B combined exceeds 70% of NAV.
- Live drawdown exceeds -10% from peak equity.
- It is a weekend or fixed US holiday (1/1, 7/4, 12/25).

Also, the code **only trades in paper unless `ALPACA_LIVE=true` is explicitly set**.

---

## 10. Monitoring

```bash
cd Live
python -m live.monitor
```

Shows:
- Latest run date.
- Account equity.
- Number of orders and turnover.
- Sleeve-level weights.

---

## 11. Tests

```bash
cd Live
python -m pytest tests/ -q
```

Smoke tests do not call Alpaca; they use cached yfinance data.

---

## 12. Expected metrics (backtest)

> **⚠️ STALE — pending re-run.** The figures below were produced BEFORE the
> `fix/live-backtest-parity` changes (10 bps sleeve turnover costs, N=2 gate
> hysteresis, removal of a `.shift` look-ahead in the sleeve backtest, and CTA
> backtest/live parity). Those fixes remove optimistic bias, so the true
> Sharpe/Calmar are expected to be **lower** than shown. Do not quote these
> numbers until the backtest is re-run and this table updated. Tracked as a
> follow-up (no backtest runner ships in this repo yet).

| Period | CAGR | Vol | Sharpe | Max DD | Calmar |
|--------|------|-----|--------|--------|--------|
| 2015-2025 | 16.5% | 7.5% | 2.20 | -7.18% | 2.30 |
| OOS 2020-2025 | 20.6% | - | 2.36 | -7.18% | 2.87 |

*Historical results do not guarantee future performance.*
