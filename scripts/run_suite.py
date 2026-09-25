#!/usr/bin/env python
"""Expand an experiment suite into individual runs, print a PLAN (what will run
vs. what will be skipped and why), then execute — skipping anything the registry
says is finished. `--dry-run` prints the plan and stops.

Usage:
    python scripts/run_suite.py configs/suite_phase1.yaml --dry-run
    python scripts/run_suite.py configs/suite_phase1.yaml
    python scripts/run_suite.py configs/suite_phase1.yaml --only A5 --model qwen2.5-3b \
        --split standard --seed 0 --runs-dir /kaggle/working/runs
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.config import ExperimentConfig, config_from_dict, load_yaml, _deep_merge  # noqa: E402
from ror.experiment import preflight, run_experiment  # noqa: E402
from ror.logging_utils import get_logger  # noqa: E402
from ror.registry import Registry  # noqa: E402

log = get_logger("ror.suite")

# arms that involve fine-tuning (seeds matter only for these)
TRAINED = {"A5", "A6", "A7", "A8"}


def expand(suite_path: str) -> tuple[list[ExperimentConfig], dict]:
    suite = load_yaml(suite_path)
    base = load_yaml(ROOT / "configs" / "base.yaml").get("defaults", {})
    arms = load_yaml(ROOT / "configs" / "arms.yaml")          # arm -> {role, supervision, inference}
    models = load_yaml(ROOT / "configs" / "models.yaml").get("models", {})  # model -> {tier, size_b, family}
    teacher = suite.get("teacher")
    phase = suite.get("phase", 1)
    overrides = suite.get("overrides", {})   # e.g. a smoke suite's small slice
    grid = suite.get("grid", {})
    keys = list(grid.keys())

    seen: set[str] = set()
    configs: list[ExperimentConfig] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        d = dict(zip(keys, combo))
        arm = d["arm"]
        model = d["model"]
        arm_spec = arms.get(arm, {})
        model_spec = models.get(model, {})

        # role/tier compatibility: student arms only with student models, etc.
        arm_role = arm_spec.get("role", "student")
        model_tier = model_spec.get("tier", "student")
        if arm_role != model_tier:
            continue

        # ≥70B models are Phase 2 (Job 4) only — never plan them on free hardware
        if phase == 1 and model_spec.get("phase2_only"):
            log.warning("dropping %s x %s: model is phase2_only (Job 4)", arm, model)
            continue

        # seeds only matter for trained arms; collapse others to seed 0
        if arm not in TRAINED and d.get("seed", 0) != 0:
            continue

        cfg_dict = _deep_merge(base, {
            **arm_spec,
            **overrides,
            "arm": arm,
            "model": model,
            "seed": d.get("seed", 0),
            "split": d.get("split", "standard"),
            "dataset": d.get("dataset", base.get("dataset", "finqa")),
            "phase": phase,
            "max_attempts": suite.get("max_attempts", 2),
        })
        if arm == "A8":
            cfg_dict["teacher"] = teacher
        cfg = config_from_dict(cfg_dict)
        if cfg.experiment_id in seen:
            continue
        seen.add(cfg.experiment_id)
        configs.append(cfg)
    return configs, suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--dry-run", action="store_true", help="plan only, do not run")
    ap.add_argument("--force", action="store_true", help="re-run even finished experiments")
    ap.add_argument("--only", help="restrict to a single arm, e.g. A5")
    ap.add_argument("--model", help="restrict to a single model key")
    ap.add_argument("--split", help="restrict to one split: standard | clean")
    ap.add_argument("--seed", type=int, help="restrict to one seed")
    ap.add_argument("--runs-dir", help="override the suite's runs_dir (e.g. on Kaggle)")
    args = ap.parse_args()

    configs, suite = expand(args.suite)
    if args.only:
        configs = [c for c in configs if c.arm == args.only]
    if args.model:
        configs = [c for c in configs if c.model == args.model]
    if args.split:
        configs = [c for c in configs if c.split == args.split]
    if args.seed is not None:
        configs = [c for c in configs if c.seed == args.seed]

    runs_dir = args.runs_dir or suite.get("runs_dir", "runs")
    reg = Registry(runs_dir, max_attempts=suite.get("max_attempts", 2))

    # --- PLAN ---
    to_run, to_skip = [], []
    for c in configs:
        do_run, reason = (True, "forced") if args.force else reg.should_run(c.experiment_id, c.resolved_name())
        if do_run:
            ready, why = preflight(c)
            if not ready:
                do_run, reason = False, f"not ready: {why}"
        (to_run if do_run else to_skip).append((c, reason))

    log.info("SUITE %s — %d experiments (%d to run, %d skipped)",
             args.suite, len(configs), len(to_run), len(to_skip))
    log.info("--- PLAN: will run ---")
    for c, reason in to_run:
        log.info("  RUN  %s  %s  (%s)", c.experiment_id, c.resolved_name(), reason)
    if to_skip:
        log.info("--- PLAN: will skip ---")
        for c, reason in to_skip:
            log.info("  SKIP %s  %s  (%s)", c.experiment_id, c.resolved_name(), reason)

    if args.dry_run:
        log.info("dry-run: nothing executed.")
        return

    # --- EXECUTE ---
    for c, _ in to_run:
        run_experiment(c, runs_dir=runs_dir, registry=reg, force=args.force)


if __name__ == "__main__":
    main()
