"""Experiment orchestration — the single entry point every run goes through.

Flow:  should_run? -> mark_running -> PLAN (logged) -> execute -> log result
       -> mark_completed  (or mark_failed on any exception)

Executing means: (train if the arm is fine-tuned) -> generate on the split ->
evaluate -> assemble a RunResult. The training/inference/eval calls delegate to
`ror.training`, `ror.inference`, `ror.data` — implement those; the flow here is
complete and is what guarantees logging + idempotency.

This shell is deliberately runnable: with the ML modules still stubbed it will
mark the run FAILED with a NotImplementedError, which is the correct, logged
behaviour until they are implemented.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig
from .logging_utils import add_run_file_handler, get_logger, remove_handler
from .registry import Registry
from .results import RunResult, append_result
from .utils import git_commit, gpu_name, set_seed

log = get_logger("ror.experiment")

# arms that require a fine-tuning step
_TRAINED_ARMS = {"A5", "A6", "A7", "A8"}


def build_plan(cfg: ExperimentConfig) -> list[str]:
    """Human-readable plan for this run — logged before anything executes."""
    steps = [f"experiment {cfg.experiment_id} :: {cfg.resolved_name()}"]
    if cfg.arm in _TRAINED_ARMS:
        steps.append(
            f"train: QLoRA {cfg.model} on {cfg.dataset} "
            f"(supervision={cfg.supervision}, epochs={cfg.epochs}, r={cfg.lora_r})"
        )
        if cfg.supervision == "pot_distilled":
            steps.append(f"  uses distilled traces from teacher={cfg.teacher}")
    else:
        steps.append(f"no training (arm {cfg.arm}, inference={cfg.inference})")
    steps.append(f"infer on split={cfg.split}, k={cfg.self_consistency_k}")
    steps.append("evaluate: exact_match, execution_accuracy, faithfulness, executability")
    steps.append("log RunResult -> runs/results.jsonl")
    return steps


def run_experiment(
    cfg: ExperimentConfig,
    runs_dir: str | Path = "runs",
    registry: Optional[Registry] = None,
    force: bool = False,
) -> Optional[RunResult]:
    """Run one experiment idempotently. Returns the RunResult, or None if skipped."""
    reg = registry or Registry(runs_dir, max_attempts=cfg.max_attempts)
    exp_id = cfg.experiment_id
    name = cfg.resolved_name()

    if not force:
        do_run, reason = reg.should_run(exp_id, name)
        if not do_run:
            log.info("SKIP  %s (%s) — %s", exp_id, name, reason)
            return None
    else:
        reg.force_reset(exp_id)

    run_dir = reg.run_dir(exp_id)
    fh = add_run_file_handler(log, run_dir)
    reg.mark_running(exp_id, name)
    t0 = time.time()

    try:
        set_seed(cfg.seed)
        log.info("PLAN for %s:", name)
        for step in build_plan(cfg):
            log.info("  - %s", step)

        # --- execute (delegates to the ML modules) ---
        result = _execute(cfg, run_dir)
        result.wall_time_s = time.time() - t0
        result.git_commit = git_commit()
        result.gpu = gpu_name()

        append_result(runs_dir, result)
        reg.mark_completed(exp_id, metrics_path=str(run_dir / "result.json"))
        log.info("DONE  %s in %.1fs — EM=%s", name, result.wall_time_s,
                 result.exact_match)
        return result

    except Exception as e:  # noqa: BLE001 — we want to log *any* failure
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        st = reg.mark_failed(exp_id, f"{type(e).__name__}: {e}")
        log.error("FAIL  %s — %s (status now: %s)", name, e, st.status)
        return None
    finally:
        remove_handler(log, fh)


def _execute(cfg: ExperimentConfig, run_dir: Path) -> RunResult:
    """Train (if needed), generate, evaluate; assemble a RunResult.

    Kept thin and import-late so the framework imports even before the ML modules
    are implemented. Implement `ror.training`, `ror.inference`, `ror.data`,
    `ror.models` to make this run for real.
    """
    from . import data, inference, models, training  # late import
    from .metrics import exact_match, execution_accuracy
    from .faithfulness import (executability_rate, program_faithfulness,
                               posthoc_proxy_agreement)

    examples = data.load_dataset(cfg.dataset, cfg.split)

    model = None
    if cfg.arm in _TRAINED_ARMS:
        model = training.train_qlora(cfg, examples_train=data.load_dataset(cfg.dataset, "train"),
                                     run_dir=run_dir)
    else:
        model = models.load_student(cfg.model) if cfg.role == "student" \
            else models.load_teacher(cfg.model, phase=cfg.phase)

    preds = inference.generate(model, cfg, examples)
    golds = [ex.answer for ex in examples]

    result = RunResult(
        exp_id=cfg.experiment_id, name=cfg.resolved_name(), arm=cfg.arm,
        model=cfg.model, role=cfg.role, supervision=cfg.supervision,
        inference=cfg.inference, dataset=cfg.dataset, split=cfg.split,
        seed=cfg.seed, phase=cfg.phase, n_examples=len(examples),
    )
    result.exact_match = exact_match([p.answer for p in preds], golds)

    if any(getattr(p, "program", None) for p in preds):
        exec_vals = [getattr(p, "exec_value", None) for p in preds]
        result.execution_accuracy = execution_accuracy(exec_vals, golds)
        result.executability_rate = executability_rate(exec_vals)
        faith_items = inference.to_faith_items(preds, golds)
        result.faithfulness_primary = program_faithfulness(faith_items)

    # post-hoc proxy for non-program arms is computed by inference.posthoc(...)
    proxy_items = inference.posthoc_faith_items(model, cfg, preds, examples)
    if proxy_items is not None:
        result.faithfulness_proxy = posthoc_proxy_agreement(proxy_items)

    result.extra["notes"] = cfg.notes
    return result
