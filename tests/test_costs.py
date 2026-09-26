"""Tests for ror.costs (per-dollar frontier)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.costs import gpu_price, load_pricing, run_cost  # noqa: E402

PRICING = {"T4": {"usd_per_gpu_hour": 0.40}, "A100": {"usd_per_gpu_hour": None}}


def _row(gpu="Tesla T4", n=1, train=3600.0, ev=1800.0):
    return {"gpu": gpu, "extra": {"gpu": {"n_gpus": n, "train_seconds": train,
                                          "eval_seconds": ev}}}


def test_price_matching_and_unpriced():
    assert gpu_price("Tesla T4", PRICING) == 0.40
    assert gpu_price("NVIDIA A100-SXM4-80GB", PRICING) is None      # listed, not priced
    assert gpu_price("NVIDIA L4", PRICING) is None and gpu_price(None, PRICING) is None


def test_run_cost_counts_gpus_and_splits_train_eval():
    assert run_cost(_row(), PRICING) == {"total": 0.6, "train": 0.4, "eval": 0.2}
    assert run_cost(_row(n=2), PRICING)["total"] == 1.2              # 32B on T4x2
    assert run_cost(_row(gpu="NVIDIA A100 80GB"), PRICING) is None   # unknown, not 0
    assert run_cost({"gpu": "Tesla T4", "extra": {}}, PRICING) is None


def test_shipped_pricing_file_parses():
    pricing = load_pricing()
    assert {"T4", "A100"} <= set(pricing)
