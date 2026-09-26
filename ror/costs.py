"""Dollar cost of runs, for the per-dollar frontier (proposal §5.6).

Runs record GPU time in `extra["gpu"]` = {"n_gpus", "train_seconds",
"eval_seconds"} (training includes every session and checkpoint selection).
Costs are computed here at analysis time from configs/pricing.yaml, so prices
can be filled in or corrected without re-running anything (roadmap P1.6, P7.3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import load_yaml
from .paths import configs_dir


def load_pricing(path: Optional[str | Path] = None) -> dict:
    """{gpu key: {"usd_per_gpu_hour", "source", "date"}} from pricing.yaml."""
    p = Path(path) if path else configs_dir() / "pricing.yaml"
    return (load_yaml(p).get("gpus") or {}) if p.exists() else {}


def gpu_price(gpu_name: Optional[str], pricing: dict) -> Optional[float]:
    """$/GPU-hour for a device name ("Tesla T4" matches key "T4"); None if unpriced."""
    if not gpu_name:
        return None
    for key, spec in pricing.items():
        if key.lower() in gpu_name.lower():
            price = (spec or {}).get("usd_per_gpu_hour")
            return float(price) if price is not None else None
    return None


def run_cost(row: dict, pricing: Optional[dict] = None) -> Optional[dict]:
    """{"total", "train", "eval"} in USD for a results.jsonl row, or None if its
    GPU time or its GPU's price is unknown. Training cost is the full cost of the
    adapter, also for evaluations that reused it (like train_flops)."""
    gpu = (row.get("extra") or {}).get("gpu") or {}
    price = gpu_price(row.get("gpu"), load_pricing() if pricing is None else pricing)
    if price is None or not gpu:
        return None
    per_second = price * int(gpu.get("n_gpus") or 1) / 3600.0
    train = per_second * float(gpu.get("train_seconds") or 0.0)
    ev = per_second * float(gpu.get("eval_seconds") or 0.0)
    return {"total": round(train + ev, 4), "train": round(train, 4), "eval": round(ev, 4)}
