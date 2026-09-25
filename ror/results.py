"""The structured result record. Every completed run appends one `RunResult` as
a JSON line to `runs/results.jsonl`. That file IS the ablation study — nothing
about a run should live only in a text log or a notebook cell.

`scripts/aggregate_results.py` reads this file to build the ablation table and
the accuracy-per-compute frontier, so keep every field populated where known.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

RESULTS_FILE = "results.jsonl"


@dataclass
class RunResult:
    # --- identity (mirror of the config) ---
    exp_id: str
    name: str
    arm: str
    model: str
    role: str = "student"
    model_family: str = ""
    model_size_b: Optional[float] = None
    supervision: str = "none"
    inference: str = "greedy"
    dataset: str = "finqa"
    split: str = "standard"          # standard | clean
    seed: int = 0
    phase: int = 1

    # --- metrics ---
    n_examples: int = 0
    exact_match: Optional[float] = None
    execution_accuracy: Optional[float] = None      # PoT arms
    faithfulness_primary: Optional[float] = None     # PoT arms
    faithfulness_proxy: Optional[float] = None        # post-hoc, all arms
    executability_rate: Optional[float] = None        # PoT arms

    # --- compute / cost (for the frontier) ---
    train_flops: Optional[float] = None
    infer_flops: Optional[float] = None
    wall_time_s: Optional[float] = None
    gpu: str = ""
    cost_usd: float = 0.0

    # --- provenance ---
    git_commit: str = ""
    timestamp: float = field(default_factory=time.time)
    config: dict[str, Any] = field(default_factory=dict)  # full ExperimentConfig
    extra: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def append_result(runs_dir: str | Path, result: RunResult) -> Path:
    """Append to the master results.jsonl and also drop a copy in the run dir."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    master = runs_dir / RESULTS_FILE
    with open(master, "a", encoding="utf-8") as f:
        f.write(result.to_line() + "\n")
    per_run = runs_dir / result.exp_id / "result.json"
    per_run.parent.mkdir(parents=True, exist_ok=True)
    per_run.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return master


def write_predictions(run_dir: str | Path, rows: list[dict]) -> Path:
    """Write one JSON line per evaluated item to <run_dir>/predictions.jsonl."""
    path = Path(run_dir) / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    return path


def load_results(runs_dir: str | Path) -> list[dict]:
    master = Path(runs_dir) / RESULTS_FILE
    if not master.exists():
        return []
    rows = []
    for line in master.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
