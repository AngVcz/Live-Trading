"""Discretionary 07:30 macro-news overlay for the A+B+Diversifier Sleeves strategy.

The LLM never touches weights. It produces only market analysis + a recommended
profile (prose + one label). The three sizing combinations are computed
deterministically from fixed regime-gate sleeve mappings (``PROFILES``), so the
strategy stays reproducible; the LLM is an advisory input the human overrides.

See ``scripts/morning_report.py`` for the end-to-end report flow and
``scripts/rebalance.py --profile`` for the pick-then-trade step.
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

# Fixed regime-gate sleeve allocations. Each sums to 1.0. The LLM picks one; it
# never edits these numbers.
PROFILES: Dict[str, Dict[str, float]] = {
    "aggressive": {"A": 0.30, "B": 0.25, "rates": 0.10, "BIL_ballast": 0.05, "cta": 0.30},
    "balanced": {"A": 0.20, "B": 0.20, "rates": 0.20, "BIL_ballast": 0.20, "cta": 0.20},
    "passive": {"A": 0.10, "B": 0.10, "rates": 0.30, "BIL_ballast": 0.25, "cta": 0.25},
}

_SLEEVE_ORDER = ["A", "B", "rates", "BIL_ballast", "cta"]
_MIKTEX_BIN = Path(r"C:\Users\angve\AppData\Local\Programs\MiKTeX\miktex\bin\x64")


def apply_profile(name: str) -> pd.Series:
    """Return the sleeve-level weight Series for a profile, renormalized to 1.0."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile '{name}'; choose one of {list(PROFILES)}")
    s = pd.Series(PROFILES[name], index=_SLEEVE_ORDER, dtype=float)
    s = s / s.sum()
    return s


def _parse_analysis(text: str) -> Dict:
    """Extract recommended profile + confidence from the model's free-form text.

    The prompt asks the model to end with two lines:
        RECOMMENDED_PROFILE: <aggressive|balanced|passive>
        CONFIDENCE: <low|med|high>
    Headlines = leading bullet lines. Everything else is the prose analysis.
    """
    rec = ""
    m = re.search(r"(?im)^\s*RECOMMENDED_PROFILE:\s*([a-zA-Z]+)\s*$", text)
    if m:
        rec = m.group(1).strip().lower()
        if rec not in PROFILES:
            rec = ""

    conf = ""
    m = re.search(r"(?im)^\s*CONFIDENCE:\s*([a-zA-Z]+)\s*$", text)
    if m:
        conf = m.group(1).strip().lower()

    headlines = re.findall(r"(?m)^\s*(?:[-*•])\s+(.+?)\s*$", text)
    # Strip the two trailing control lines from the prose body.
    body = re.sub(r"(?im)^\s*RECOMMENDED_PROFILE:.*$\n?", "", text)
    body = re.sub(r"(?im)^\s*CONFIDENCE:.*$\n?", "", body)
    body = body.strip()

    return {
        "market_analysis": body,
        "recommended_profile": rec,
        "confidence": conf,
        "rationale": "",  # prose body already carries the rationale
        "headlines": headlines[:5],
    }


def analyze(run_date: date) -> Dict:
    """Call the headless ``claude`` CLI for market analysis + a recommended profile.

    Reuses the user's existing Claude Code auth (no extra key). On any failure the
    report still ships deterministic tables with ``source == "none"``.
    """
    prompt = (
        "You are a macro/market analyst for a dual-momentum ETF portfolio with five "
        "sleeves: equity-momentum engines A and B, a rates sleeve (TLT/IEF/BIL by "
        "200-day trend), a BIL ballast sleeve (cash proxy; replaces the SH bear sleeve), "
        "and a CTA proxy (PDBC/DBMF/KMLM in uptrend, else BIL).\n\n"
        f"Today is {run_date.isoformat()}. Use web search to gather the latest "
        "financial and macroeconomic news: central banks (Fed/ECB), inflation/CPI, "
        "employment, GDP, geopolitics, equity-market moves, VIX, Treasury yields, "
        "and commodities.\n\n"
        "Write a concise market analysis (3-6 short paragraphs). Then recommend "
        "exactly ONE of three fixed sizing profiles for today:\n"
        "  - aggressive: risk-on tilt (more A/B equity momentum and CTA, less BIL ballast)\n"
        "  - balanced:   neutral, 20% in each sleeve (the systematic baseline)\n"
        "  - passive:    risk-off tilt (more rates/BIL ballast/defensive, less equity)\n\n"
        "Optionally prefix your response with up to 5 one-line headline bullets "
        "(each line starting with '- '). End your response with exactly two lines:\n"
        "RECOMMENDED_PROFILE: <aggressive|balanced|passive>\n"
        "CONFIDENCE: <low|med|high>\n"
        "Do not add anything after those two lines."
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "WebSearch"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if proc.returncode != 0:
            return _no_analysis(f"claude exit {proc.returncode}")
        env = json.loads(proc.stdout)
        if env.get("is_error") or env.get("subtype") != "success":
            return _no_analysis(env.get("result") or "claude error")
        parsed = _parse_analysis(env.get("result", ""))
        parsed["source"] = "claude-cli"
        return parsed
    except Exception as e:  # ponytail: any failure -> deterministic-only report
        return _no_analysis(str(e))


def _no_analysis(reason: str) -> Dict:
    return {
        "market_analysis": f"(Market analysis unavailable: {reason}. Sizing tables below are still valid.)",
        "recommended_profile": "",
        "confidence": "",
        "rationale": "",
        "headlines": [],
        "source": "none",
    }


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
    for k in _SLEEVE_ORDER:
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


def build_report(
    run_date: date,
    systematic: Dict,
    profiles: Dict[str, Dict],
    analysis: Dict,
    equity: float,
    out_pdf: Path,
) -> Path:
    """Render the LaTeX report to ``out_pdf`` via pdflatex. Returns the PDF path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    build_dir = LOG_DIR / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    rec = analysis.get("recommended_profile", "")
    conf = analysis.get("confidence", "")
    body = analysis.get("market_analysis", "")

    sections = []
    sections.append(f"\\section*{{Original systematic sizings}}\n"
                    f"Account equity: \\${equity:,.0f}\n\n"
                    f"\\subsection*{{Sleeve}}\n{_sleeve_table(systematic['sleeve'], equity)}\n\n"
                    f"\\subsection*{{Tickers}}\n{_ticker_table(systematic['tickers'], equity)}")

    sections.append("\\section*{Market analysis}\n" + _tex_escape(body).replace("\n", "\n\n"))
    sections.append(f"\\section*{{Recommendation}}\n"
                    f"Recommended profile: \\textbf{{{_tex_escape(rec or 'none')}}} "
                    f"(confidence: {_tex_escape(conf or 'n/a')})")

    for name in PROFILES:
        p = profiles[name]
        star = " \\textbf{(recommended)}" if name == rec else ""
        sections.append(f"\\section*{{Sizing combination: {name}{star}}}\n"
                        f"\\subsection*{{Sleeve}}\n{_sleeve_table(p['sleeve'], equity)}\n\n"
                        f"\\subsection*{{Tickers}}\n{_ticker_table(p['tickers'], equity)}")

    sections.append("\\section*{How to execute}\n"
                    "Pick one profile and run:\\\\\n"
                    f"\\texttt{{python scripts/rebalance.py --profile "
                    f"{rec or '<name>'} --date {run_date.isoformat()}}}")

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
            capture_output=True, text=True, timeout=120,
        )
        built = build_dir / tex_path.with_suffix(".pdf").name
        if built.exists():
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built), str(out_pdf))
            return out_pdf
    # ponytail: no pdflatex -> leave the .tex so the user can compile by hand
    print(f"WARN: pdflatex not found or failed; .tex left at {tex_path}")
    return tex_path


# ---- self-check ------------------------------------------------------------

def _self_check() -> None:
    # 1. profiles sum to 1.0 and apply_profile renormalizes.
    for name, vec in PROFILES.items():
        assert abs(sum(vec.values()) - 1.0) < 1e-9, f"{name} does not sum to 1.0"
        s = apply_profile(name)
        assert list(s.index) == _SLEEVE_ORDER, f"{name} wrong index"
        assert abs(s.sum() - 1.0) < 1e-9, f"{name} not renormalized to 1.0"
    # 2. parser extracts the control lines and leaves prose.
    sample = (
        "- Fed held rates, signaled patience.\n"
        "- CPI cooled to 2.9%.\n\n"
        "Equity momentum is intact but valuations are stretched. Yields fell, "
        "supporting the rates sleeve. VIX is elevated.\n\n"
        "RECOMMENDED_PROFILE: balanced\n"
        "CONFIDENCE: med\n"
    )
    parsed = _parse_analysis(sample)
    assert parsed["recommended_profile"] == "balanced", parsed
    assert parsed["confidence"] == "med", parsed
    assert "RECOMMENDED_PROFILE" not in parsed["market_analysis"]
    assert len(parsed["headlines"]) == 2, parsed
    # 3. unknown recommendation is rejected.
    bad = _parse_analysis("analysis\nRECOMMENDED_PROFILE: moonshot\nCONFIDENCE: high\n")
    assert bad["recommended_profile"] == "", bad
    print("discretionary self-check OK: 3 profiles, parser, apply_profile")


if __name__ == "__main__":
    _self_check()