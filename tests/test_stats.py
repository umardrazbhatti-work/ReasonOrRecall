"""Tests for ror.stats (bootstrap CIs, McNemar, Holm)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.stats import (bootstrap_diff_ci, bootstrap_mean_ci, holm, item_scores,  # noqa: E402
                       mcnemar_exact, seed_sd)


def test_bootstrap_ci_brackets_the_mean_and_matches_theory():
    x = [1] * 200 + [0] * 200                       # accuracy 0.5, n = 400
    mean, lo, hi = bootstrap_mean_ci(x, seed=1)
    assert mean == 0.5 and lo < 0.5 < hi
    half = 1.96 * math.sqrt(0.25 / 400)              # normal approximation: 0.049
    assert abs((hi - lo) / 2 - half) < 0.006
    assert bootstrap_mean_ci(x, seed=1) == (mean, lo, hi)   # deterministic


def test_gap_ci_unpaired_matches_the_roadmap_power_estimate():
    # roadmap §5: 1,147 standard vs 400 clean items at 50% -> gap SE ~2.9 points
    std, clean = [1, 0] * 573 + [1], [1, 0] * 200
    gap, lo, hi = bootstrap_diff_ci(std, clean, paired=False, seed=2)
    assert abs(gap) < 0.01
    assert abs((hi - lo) / 2 - 1.96 * 0.029) < 0.008


def test_paired_diff_uses_the_same_items():
    a = [1, 1, 1, 0] * 50
    b = [1, 0, 1, 0] * 50
    diff, lo, hi = bootstrap_diff_ci(a, b, paired=True, seed=3)
    assert diff == 0.25 and 0.15 < lo < diff < hi < 0.35
    with pytest.raises(ValueError):
        bootstrap_diff_ci([1, 0], [1], paired=True)


def test_mcnemar_exact():
    r = mcnemar_exact([1] * 10 + [0] * 90, [0] * 100)       # 10 discordant, all one way
    assert (r["only_a"], r["only_b"]) == (10, 0) and r["p"] == pytest.approx(2 / 1024)
    assert mcnemar_exact([1, 0], [1, 0])["p"] == 1.0          # no discordant pairs
    assert mcnemar_exact([1, 0, 1, 0], [0, 1, 0, 1])["p"] == 1.0


def test_holm_adjustment():
    adj = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert adj == {"H1": 0.03, "H3": 0.06, "H2": 0.06}       # monotone, capped at 1


def test_seed_sd_and_item_scores():
    assert seed_sd([0.40, 0.44]) == pytest.approx(0.028284, rel=1e-4)
    assert seed_sd([0.4]) is None
    scores = item_scores([{"a": True, "b": False, "c": True}, {"a": True, "b": True}])
    assert scores == {"a": 1.0, "b": 0.5}                     # only items in every seed
