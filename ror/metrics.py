"""Answer normalization and the core accuracy metrics.

FinQA-style answers are numeric (often with $, %, commas, or "in millions"), so
Exact Match must normalize before comparing, with a tolerance for floats. String
answers fall back to a normalized string compare.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

_SCALE = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mm": 1e6, "mn": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9,
    "trillion": 1e12, "trillions": 1e12,
}

# optional sign, then "12", "12.", "12.5" or ".5", then an optional exponent
_NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?")


def normalize_number(text) -> Optional[float]:
    """Parse a numeric answer to a float, or return None if not numeric.

    Handles: $ signs, thousands commas, percentages (kept as the printed number,
    e.g. "15%" -> 15.0), parentheses-negatives "(1,200)" -> -1200, and a trailing
    scale word ("1.2 million" -> 1_200_000).
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().lower().replace("−", "-")  # unicode minus
    if not s:
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    scale = 1.0
    for word, mult in _SCALE.items():
        if re.search(rf"\b{re.escape(word)}\b", s):
            scale = mult
            s = re.sub(rf"\b{re.escape(word)}\b", "", s)
            break

    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    val = float(m.group()) * scale
    return -val if negative else val


def answers_match(pred, gold, rel_tol: float = 1e-3, abs_tol: float = 1e-4) -> bool:
    """True if pred equals gold: numeric within tolerance, else normalized string."""
    gp, gg = normalize_number(pred), normalize_number(gold)
    if gp is not None and gg is not None:
        denom = max(abs(gg), 1.0)
        return abs(gp - gg) <= max(abs_tol, rel_tol * denom)
    return _norm_str(pred) == _norm_str(gold)


def _norm_str(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower()).strip(" .")


def exact_match(preds: Sequence, golds: Sequence, **kw) -> float:
    """Fraction of predictions matching gold. preds/golds align by index."""
    if not golds:
        return 0.0
    hits = sum(answers_match(p, g, **kw) for p, g in zip(preds, golds))
    return hits / len(golds)


def execution_accuracy(exec_values: Sequence, golds: Sequence, **kw) -> float:
    """For PoT arms: fraction where the *executed program value* matches gold.
    `exec_values` may contain None for programs that failed to run."""
    if not golds:
        return 0.0
    hits = 0
    for v, g in zip(exec_values, golds):
        if v is not None and answers_match(v, g, **kw):
            hits += 1
    return hits / len(golds)
