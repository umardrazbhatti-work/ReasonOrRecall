"""Statistics for the study's claims (roadmap P1.5, docs/PROTOCOL.md).

- `bootstrap_mean_ci`: 95% percentile-bootstrap CI of an accuracy (resampling
  test items). With several seeds, pass each item's mean correctness across
  seeds; seed-to-seed spread is reported separately (`seed_sd`).
- `bootstrap_diff_ci`: CI of a difference of two accuracies. `paired=True` for
  two systems on the same items (arm vs arm); `paired=False` for different item
  sets (standard vs clean: the contamination gap).
- `mcnemar_exact`: exact McNemar test for two systems on the same items.
- `holm`: Holm-Bonferroni adjusted p-values for a family of hypotheses.

Everything is seeded, so the same inputs always give the same intervals.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

N_BOOT = 10_000
ALPHA = 0.05


def _arr(x: Sequence) -> np.ndarray:
    return np.asarray([float(v) for v in x], dtype=float)


def bootstrap_mean_ci(x: Sequence, n_boot: int = N_BOOT, alpha: float = ALPHA,
                      seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) of the per-item scores `x` (0/1 or fractions)."""
    a = _arr(x)
    if a.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(a.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a: Sequence, b: Sequence, paired: bool, n_boot: int = N_BOOT,
                      alpha: float = ALPHA, seed: int = 0) -> tuple[float, float, float]:
    """(mean(a) - mean(b), low, high). Paired: a[i] and b[i] are the same item
    and are resampled together; unpaired: each set is resampled on its own."""
    x, y = _arr(a), _arr(b)
    if x.size == 0 or y.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    if paired:
        if x.size != y.size:
            raise ValueError("paired comparison needs the same items in both")
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    else:
        diffs = (x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
                 - y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(x.mean() - y.mean()), float(lo), float(hi)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired correctness. Returns the
    discordant counts (only_a, only_b) and the p-value."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired comparison needs the same items in both")
    only_a = sum(bool(x) and not bool(y) for x, y in zip(a_correct, b_correct))
    only_b = sum(bool(y) and not bool(x) for x, y in zip(a_correct, b_correct))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail)}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (same keys)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def seed_sd(values: Sequence[Optional[float]]) -> Optional[float]:
    """Sample standard deviation across seeds (None with fewer than two)."""
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def item_scores(runs_predictions: Sequence[Mapping[str, bool]]) -> dict[str, float]:
    """Per-item mean correctness across seeds: {uid: fraction of seeds correct},
    over the items present in every seed's predictions."""
    if not runs_predictions:
        return {}
    common = set(runs_predictions[0])
    for preds in runs_predictions[1:]:
        common &= set(preds)
    return {uid: sum(bool(p[uid]) for p in runs_predictions) / len(runs_predictions)
            for uid in sorted(common)}
