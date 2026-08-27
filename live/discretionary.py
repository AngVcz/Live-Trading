"""Two-stage discretionary 07:30 macro-news overlay for the A+B+Diversifier
Sleeves strategy. The LLM never touches weights: Stage 1 emits an executive
summary, Stage 2 a committee ranking; all sizings come from deterministic tilt
rules. See prompts/01_news_exec_summary.md and prompts/02_options_analysis.md.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"

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
    if abs(float(sleeve.get("bear", 0.0))) >= 1e-9:
        # Not a bare assert: this guards traded weights and must survive `python -O`.
        raise ValueError("tilt engine assumes the bear sleeve is disabled")
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
    {"sleeve": {...}, "tickers": {...}, "note": str} with weights summing to 1.0
    for non-degenerate inputs (sum > 0).
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
        raise RuntimeError(f"claude exit {proc.returncode}: {(proc.stderr or '').strip()[:200]}")
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
    # Models sometimes emit the typographic apostrophe (U+2019) in this header.
    macro_calendar = re.findall(r"(?m)^\s*[-*•]\s+(.+?)\s*$",
                                _section(text, "Today's events")
                                or _section(text, "Today’s events"))
    return {
        "exec_summary": _strip_control_lines(text, ("REGIME_BIAS", "SUMMARY_CONFIDENCE")),
        "headlines": re.findall(r"(?m)^\s*[-*•]\s+(.+?)\s*$",
                                _section(text, "Headlines"))[:5],
        "macro_calendar": macro_calendar,
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
    try:  # render, CLI call, and parse all degrade on ANY failure
        prompt = _render_prompt("01_news_exec_summary.md",
                                DATE=run_date.isoformat(),
                                METRICS_JSON=json.dumps(metrics, indent=2, default=str))
        parsed = _parse_stage1(_call_claude(prompt, model))
        parsed["source"] = "claude-cli"
        return parsed
    except Exception as e:  # ponytail: any failure -> deterministic-only report
        return _no_stage1(str(e))


def stage2_decide(run_date: date, stage1: Dict, options: Dict[str, Dict],
                  model: str | None = None) -> Dict:
    """Stage 2: committee ranking of the three options per prompts/02_options_analysis.md."""
    try:  # render, CLI call, and parse all degrade on ANY failure
        prompt = _render_prompt("02_options_analysis.md",
                                DATE=run_date.isoformat(),
                                STAGE1_TEXT=stage1.get("exec_summary", "(unavailable)"),
                                # _parse_stage1 strips these labels out of
                                # exec_summary; pass them so the committee can see
                                # the self-report (veto trigger: low confidence).
                                STAGE1_BIAS=stage1.get("regime_bias") or "unavailable",
                                STAGE1_CONF=stage1.get("summary_confidence") or "unavailable",
                                OPTIONS_JSON=json.dumps(options, indent=2, default=str))
        parsed = _parse_stage2(_call_claude(prompt, model))
        parsed["source"] = "claude-cli"
        return parsed
    except Exception as e:
        return _no_stage2(str(e))


# Fixed regime-gate sleeve allocations. Each sums to 1.0. The LLM picks one; it
# never edits these numbers.
PROFILES: Dict[str, Dict[str, float]] = {
    "aggressive": {"A": 0.30, "B": 0.25, "rates": 0.10, "BIL_ballast": 0.05, "cta": 0.30},
    "balanced": {"A": 0.20, "B": 0.20, "rates": 0.20, "BIL_ballast": 0.20, "cta": 0.20},
    "passive": {"A": 0.10, "B": 0.10, "rates": 0.30, "BIL_ballast": 0.25, "cta": 0.25},
}

_MIKTEX_BIN = Path(r"C:\Users\angve\AppData\Local\Programs\MiKTeX\miktex\bin\x64")


def apply_profile(name: str) -> pd.Series:
    """Return the sleeve-level weight Series for a profile, renormalized to 1.0."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile '{name}'; choose one of {list(PROFILES)}")
    s = pd.Series(PROFILES[name], index=_SLEEVE_ORDER, dtype=float)
    s = s / s.sum()
    return s


# ---- LaTeX report ----------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


def _pct(w: float) -> str:
    return f"{w * 100:.1f}"


def _dollars(w: float, equity: float) -> str:
    return f"{w * equity:,.0f}"


def _sleeve_table(sleeve: Dict[str, float], equity: float) -> str:
    rows = []
    for k in SLEEVE_ORDER:
        w = sleeve.get(k, 0.0)
        rows.append(f"{_tex_escape(k)} & {_pct(w)} & {_dollars(w, equity)} \\\\")
    return (
        "\\begin{tabular}{lrr}\n\\hline\n"
        "Sleeve & Weight (\\%) & \\\\\n\\hline\n"
        + "\n".join(rows) + "\n\\hline\n\\end{tabular}"
    )


def _ticker_table(tickers: Dict[str, float], equity: float) -> str:
    rows = []
    for t, w in sorted(tickers.items(), key=lambda kv: kv[1], reverse=True):
        rows.append(f"{_tex_escape(t)} & {_pct(w)} & {_dollars(w, equity)} \\\\")
    return (
        "\\begin{tabular}{lrr}\n\\hline\n"
        "Ticker & Weight (\\%) & \\\\\n\\hline\n"
        + "\n".join(rows) + "\n\\hline\n\\end{tabular}"
    )


def _find_pdflatex() -> str | None:
    found = shutil.which("pdflatex")
    if found:
        return found
    candidate = _MIKTEX_BIN / "pdflatex.exe"
    return str(candidate) if candidate.exists() else None


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
            f"\\section*{{Option: {_tex_escape(name)}}}\n"
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
        f"{_tex_escape(stage2.get('recommended_option') or '<name>')} "
        f"--date {run_date.isoformat()}}}")

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


# ---- self-check ------------------------------------------------------------

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