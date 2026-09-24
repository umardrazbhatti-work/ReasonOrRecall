#!/usr/bin/env python
"""Run a single experiment from a config YAML (merged over configs/base.yaml).

Usage:
    python scripts/run_experiment.py configs/experiments/A5_qlora_answer.yaml
    python scripts/run_experiment.py configs/experiments/A5_qlora_answer.yaml --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.config import load_experiment_config  # noqa: E402
from ror.experiment import run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config, base_yaml=ROOT / "configs" / "base.yaml")
    run_experiment(cfg, runs_dir=args.runs_dir, force=args.force)


if __name__ == "__main__":
    main()
