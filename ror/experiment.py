"""Experiment orchestration — the single entry point every run goes through.

Flow:  should_run? -> mark_running -> PLAN (logged) -> execute -> log result
       -> mark_completed  (or mark_failed on any exception)

Executing means: (train if the arm is fine-tuned) -> generate on the split ->
evaluate -> assemble a RunResult. The training/inference/eval calls delegate to
`ror.training`, `ror.inference`, `ror.data` — implement those; the flow here is
complete and is what guarantees logging + idempotency.

This shell is deliberately runnable: with the ML modules still stubbed, a run
logs the NotImplementedError (runs/<id>/error.txt) and returns to PENDING
*without* consuming an attempt — an unimplemented stub is not an experiment
failure, and counting it would leave every experiment permanently failed before
the code exists. NotImplementedError raised by a library still counts.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig
from .logging_utils import add_run_file_handler, get_logger, remove_handler
from .registry import Registry
from .results import RunResult, append_result, write_predictions
from .utils import (RunPaused, git_commit, gpu_name, inference_flops, library_versions,
                    set_seed)

log = get_logger("ror.experiment")

# arms that require a fine-tuning step
_TRAINED_ARMS = {"A5", "A6", "A7", "A8"}


def build_plan(cfg: ExperimentConfig) -> list[str]:
    """Human-readable plan for this run — logged before anything executes."""
    steps = [f"experiment {cfg.experiment_id} :: {cfg.resolved_name()}"]
    if cfg.arm in _TRAINED_ARMS:
        steps.append(
            f"train: QLoRA {cfg.model} on {cfg.dataset} "
            f"(supervision={cfg.supervision}, epochs={cfg.epochs}, r={cfg.lora_r}, "
            f"max_seq_len={cfg.max_seq_len}, "
            f"examples={cfg.train_examples or 'all'})"
        )
        if cfg.supervision == "pot_distilled":
            steps.append(f"  uses distilled traces from teacher={cfg.teacher}")
    else:
        steps.append(f"no training (arm {cfg.arm}, inference={cfg.inference})")
    steps.append(f"infer on split={cfg.split} ({cfg.eval_examples or 'all'} examples), "
                 f"k={cfg.self_consistency_k}")
    steps.append("evaluate: exact_match, execution_accuracy, faithfulness, executability")
    steps.append("log RunResult -> runs/results.jsonl")
    return steps


def preflight(cfg: ExperimentConfig) -> tuple[bool, str]:
    """Conditions that must hold before a run may consume an attempt (the data it
    needs exists). Returns (ok, reason-if-not)."""
    from .data import split_available

    needed = [cfg.split] + (["train"] if cfg.arm in _TRAINED_ARMS else [])
    for split in needed:
        ok, why = split_available(cfg.dataset, split)
        if not ok:
            return False, why
    return True, ""


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

    ready, why = preflight(cfg)
    if not ready:
        log.warning("SKIP  %s (%s) — not ready: %s", exp_id, name, why)
        return None
    if force:
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
        result.config = asdict(cfg)

        append_result(runs_dir, result)
        reg.mark_completed(exp_id, metrics_path=str(run_dir / "result.json"))
        log.info("DONE  %s in %.1fs — EM=%s (strict %s)", name, result.wall_time_s,
                 result.exact_match, result.exact_match_strict)
        return result

    except Exception as e:  # noqa: BLE001 — we want to log *any* failure
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb, encoding="utf-8")
        if isinstance(e, RunPaused):
            reg.mark_not_ready(exp_id, f"paused: {e}")
            log.warning("PAUSED  %s — %s (attempt not counted; resumes next session)",
                        name, e)
            return None
        if _raised_by_stub(e):
            reg.mark_not_ready(exp_id, f"{type(e).__name__}: {e}")
            log.warning("NOT READY  %s — %s (attempt not counted; status: pending)",
                        name, e)
            return None
        st = reg.mark_failed(exp_id, f"{type(e).__name__}: {e}")
        log.error("FAIL  %s — %s (status now: %s)", name, e, st.status)
        return None
    finally:
        remove_handler(log, fh)


def _raised_by_stub(exc: BaseException) -> bool:
    """True if `exc` is a NotImplementedError raised directly by code in the
    `ror` package (an unimplemented stub), as opposed to one from a library."""
    if not isinstance(exc, NotImplementedError):
        return False
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    origin = Path(tb.tb_frame.f_code.co_filename).resolve().parent
    return origin == Path(__file__).resolve().parent


def _execute(cfg: ExperimentConfig, run_dir: Path) -> RunResult:
    """Train (if needed), generate, evaluate; assemble a RunResult.

    Kept thin and import-late so the framework imports even before the ML modules
    are implemented. Implement `ror.training`, `ror.inference`, `ror.data`,
    `ror.models` to make this run for real.
    """
    from . import data, inference, models, training  # late import
    from .metrics import (exact_match, execution_accuracy, primary_exact_match,
                          primary_execution_accuracy)
    from .faithfulness import (executability_rate, program_faithfulness,
                               posthoc_proxy_agreement)

    examples = data.load_dataset(cfg.dataset, cfg.split)
    if cfg.eval_examples:
        examples = examples[:cfg.eval_examples]

    t0 = time.time()
    model = None
    if cfg.arm in _TRAINED_ARMS:
        model = training.train_qlora(cfg, examples_train=data.load_dataset(cfg.dataset, "train"),
                                     run_dir=run_dir)
    else:
        model = models.load_student(cfg.model) if cfg.role == "student" \
            else models.load_teacher(cfg.model, phase=cfg.phase)
    t1 = time.time()

    preds = inference.generate(model, cfg, examples)
    t2 = time.time()
    golds = [ex.answer for ex in examples]

    spec = models.resolve_model(cfg.model)
    result = RunResult(
        exp_id=cfg.experiment_id, name=cfg.resolved_name(), arm=cfg.arm,
        model=cfg.model, role=cfg.role, supervision=cfg.supervision,
        inference=cfg.inference, dataset=cfg.dataset, split=cfg.split,
        seed=cfg.seed, phase=cfg.phase, n_examples=len(examples),
        model_family=spec.get("family", ""), model_size_b=spec.get("size_b"),
    )
    answers = [p.answer for p in preds]
    result.exact_match = primary_exact_match(answers, golds, [p.percent_form for p in preds])
    result.exact_match_strict = exact_match(answers, golds)
    _record_compute(result, model, preds, float(spec.get("size_b") or 0.0) * 1e9)
    write_predictions(run_dir, _prediction_rows(preds, golds))

    if any(getattr(p, "program", None) for p in preds):
        exec_vals = [getattr(p, "exec_value", None) for p in preds]
        result.execution_accuracy = primary_execution_accuracy(exec_vals, golds)
        result.extra["execution_accuracy_strict"] = execution_accuracy(exec_vals, golds)
        result.executability_rate = executability_rate(exec_vals)
        faith_items = inference.to_faith_items(preds, golds)
        result.faithfulness_primary = program_faithfulness(faith_items)

    # post-hoc proxy for non-program arms is computed by inference.posthoc(...)
    proxy_items = inference.posthoc_faith_items(model, cfg, preds, examples)
    if proxy_items is not None:
        result.faithfulness_proxy = posthoc_proxy_agreement(proxy_items)

    result.extra["notes"] = cfg.notes
    result.extra["data"] = data.dataset_fingerprint(cfg.dataset, cfg.split)
    result.extra["unparsed_answers"] = sum(p.answer is None for p in preds)
    result.extra["decoding"] = inference.decoding_settings(cfg)
    result.extra["timing_s"] = {"model_and_training": round(t1 - t0, 1),
                                "inference": round(t2 - t1, 1)}
    result.extra["versions"] = library_versions()
    return result


def _record_compute(result: RunResult, model: object, preds: list, n_params: float) -> None:
    """Training FLOPs/stats from the trained model, inference FLOPs from the
    actual prompt and generated token counts (for the compute frontier)."""
    stats = getattr(model, "train_stats", None) or {}
    if stats:
        result.train_flops = stats.get("train_flops")
        result.extra["train"] = stats
    if preds:
        n = len(preds)
        result.infer_flops = inference_flops(
            n_params, prompt_tokens=sum(p.prompt_tokens for p in preds) / n,
            gen_tokens=sum(p.gen_tokens for p in preds) / n, n_examples=n)


def _prediction_rows(preds: list, golds: list) -> list[dict]:
    """Per-item records for runs/<exp_id>/predictions.jsonl (error analysis and
    the per-item reason-vs-recall analysis)."""
    from .metrics import answers_match, primary_match

    return [{"uid": p.uid, "gold": g, "pred": p.answer,
             "correct": primary_match(p.answer, g, p.percent_form),
             "correct_strict": answers_match(p.answer, g), "percent_form": p.percent_form,
             "text": p.text, "program": p.program, "exec_value": p.exec_value,
             "prompt_tokens": p.prompt_tokens, "gen_tokens": p.gen_tokens}
            for p, g in zip(preds, golds)]
