"""QLoRA supervised fine-tuning (Job 1).

**Train once, evaluate on every split** (roadmap P1.1): an adapter is identified
by `cfg.training_id` (training fields only, no split) and lives in
`<runs>/_adapters/<training_id>/`, shared by every experiment that differs only
in its evaluation split or decoding. The first such experiment trains; the
others reuse the adapter.

`train_qlora` is **resumable**: checkpoints go to the training folder every
SAVE_STEPS optimizer steps and training resumes from the latest one. It also
respects the session time budget ($ROR_DEADLINE_UNIX, set by the Kaggle
notebook): shortly before the deadline it saves, stops, and raises RunPaused so
the next session continues instead of being hard-killed mid-step.

Outputs in <runs>/_adapters/<training_id>/:
  adapter/          the trained LoRA adapter (the product; checkpoints are
                    deleted once it is saved)
  train_stats.json  counts, losses, throughput, estimated FLOPs, versions
Each experiment's run folder gets training.json pointing at it.
"""
from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .config import ExperimentConfig
from .data import PROMPT_STYLE, Example, build_messages, format_target, load_dataset
from .logging_utils import get_logger
from .models import LoadedModel, compute_dtype, load_student, use_cuda
from .utils import RunPaused, library_versions, seconds_left, training_flops

log = get_logger("ror.training")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAVE_STEPS = 50                   # ~20 min of T4 time for the full A5 run
WARMUP_FRACTION = 0.03            # of total optimizer steps (linear warmup, then cosine)
TRAIN_DONE = "train_done.json"    # training finished; checkpoint selection pending
SELECTION_DIR = "selection"       # adapter copies at the selection steps (step-N/)
LOG_POINTS = 50                   # training-curve points logged per run (for the report)
ADAPTERS_DIR = "_adapters"        # <runs>/_adapters/<training_id>/ (no status.json: not an experiment)
TRAIN_STOP_RESERVE_S = 45 * 60    # stop training this long before the deadline
MIN_SELECTION_S = 35 * 60         # after training, pause before selection if less is left
MIN_INFERENCE_S = 20 * 60         # after selection, pause before inference if less is left


def train_qlora(
    cfg: ExperimentConfig,
    examples_train: list[Example],
    run_dir: str | Path,
) -> LoadedModel:
    """Fine-tune a student with QLoRA under cfg.supervision and return it, loaded
    with the selected adapter and ready for `ror.inference.generate`.

    Two stages, each resumable across Kaggle sessions:
      1. training (`_train`): 1 epoch by default; a copy of the adapter is kept
         at `cfg.selection_checks` evenly spaced steps; a TRAIN_DONE marker
         records the finished training;
      2. checkpoint selection (`_select_checkpoint`), the proposal's early
         stopping (roadmap M2): every kept checkpoint answers the dev slice
         through the same `generate` as the test set; the best by primary EM
         (then faithfulness, then the earlier step) becomes the adapter.

    The adapter lives in `training_dir(cfg, run_dir)`: a finished one is reused
    (another split of the same training, or a retry after a later step failed).
    """
    run_dir = Path(run_dir)
    home = training_dir(cfg, run_dir)
    adapter_dir = home / "adapter"
    stats_path = home / "train_stats.json"
    if (adapter_dir / "adapter_config.json").exists() and stats_path.exists():
        log.info("adapter %s already trained, reusing it", cfg.training_id)
        _link(run_dir, home, cfg, reused=True)
        return _load_trained(cfg, adapter_dir, stats_path, reused=True)

    done_path = home / TRAIN_DONE
    if done_path.exists():
        stats = json.loads(done_path.read_text(encoding="utf-8"))
        log.info("training %s finished in an earlier session; selecting the checkpoint",
                 cfg.training_id)
    else:
        _check_budget(TRAIN_STOP_RESERVE_S, "not enough session time left to train")
        home.mkdir(parents=True, exist_ok=True)
        stats = _train(cfg, examples_train, home)          # raises RunPaused near the deadline
        done_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _check_budget(MIN_SELECTION_S, "trained; not enough session time left to select "
                                   "the checkpoint")
    stats["selection"] = _select_checkpoint(cfg, home)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.rmtree(home / SELECTION_DIR, ignore_errors=True)   # the chosen one is in adapter/
    done_path.unlink(missing_ok=True)
    _link(run_dir, home, cfg, reused=False)
    _check_budget(MIN_INFERENCE_S, "trained; not enough session time left for inference")
    return _load_trained(cfg, adapter_dir, stats_path, reused=False)


def selection_steps(total_steps: int, checks: int) -> list[int]:
    """Evenly spaced optimizer steps at which a checkpoint is kept for
    selection; the last step is always one of them."""
    interval = max(1, math.ceil(total_steps / max(1, checks)))
    return sorted({min(k * interval, total_steps) for k in range(1, max(1, checks) + 1)})


def _train(cfg: ExperimentConfig, examples_train: list[Example], home: Path) -> dict:
    """Run (or resume) the QLoRA training in `home`; keep an adapter copy at each
    selection step in home/selection/step-N; return the training stats."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    base = load_student(cfg.model)
    tok = base.tokenizer
    tok.padding_side = "right"
    train_recs, n_dropped, tokens_per_epoch = build_sft_records(cfg, examples_train, tok)
    dev_recs, _, _ = build_sft_records(cfg, _dev_examples(cfg), tok, sample=False)
    log.info("training records: %d (dropped %d longer than %d tokens), %d tokens/epoch; "
             "dev records: %d", len(train_recs), n_dropped, cfg.max_seq_len,
             tokens_per_epoch, len(dev_recs))
    if not train_recs:
        raise ValueError("no training records after filtering")

    model = base.model
    if base.quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))

    dtype = compute_dtype()
    ckpt_dir = home / "checkpoints"
    steps_per_epoch = math.ceil(len(train_recs) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    checks = selection_steps(total_steps, cfg.selection_checks)
    log.info("checkpoint selection at steps %s of %d", checks, total_steps)
    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(WARMUP_FRACTION * total_steps),
        logging_steps=max(1, total_steps // LOG_POINTS),   # logging only; not part of the id
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no",           # dev loss is evaluated at the selection steps (callback)
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=cfg.batch_size > 1,   # less padding in multi-item batches
        fp16=use_cuda() and dtype == torch.float16,
        bf16=use_cuda() and dtype == torch.bfloat16,
        use_cpu=not use_cuda(),
        optim="paged_adamw_8bit" if base.quantized else "adamw_torch",
        seed=cfg.seed,
        data_seed=cfg.seed,
        max_length=cfg.max_seq_len,
        completion_only_loss=True,   # loss on the answer only, not the long report
        packing=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    _single_device(args)
    budget = _time_budget_callback()
    keeper = _selection_callback(set(checks), home / SELECTION_DIR, evaluate=bool(dev_recs))
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tok,
        train_dataset=Dataset.from_list(train_recs),
        eval_dataset=Dataset.from_list(dev_recs) if dev_recs else None,
        callbacks=[budget, keeper],
    )
    n_cast = _adapters_fp32(trainer.model)
    if n_cast:
        log.info("LoRA weights kept in fp32 (%d tensors cast back from trl's bf16)", n_cast)
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    start_step = _checkpoint_step(last)
    log.info("QLoRA: %s", f"resuming from {last}" if last else "starting fresh")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=last)
    runtime = time.time() - t0
    if budget.paused:
        raise RunPaused(f"session time budget reached at step {trainer.state.global_step}"
                        f"/{trainer.state.max_steps}; resumes from {ckpt_dir}")
    if not math.isfinite(out.training_loss):
        raise RuntimeError(f"training loss is {out.training_loss}")

    steps_now = trainer.state.global_step - start_step
    tokens_per_step = tokens_per_epoch * cfg.epochs / max(trainer.state.max_steps, 1)
    stats = {
        "n_train": len(train_recs), "n_dropped_too_long": n_dropped,
        "n_dev": len(dev_recs), "train_tokens_per_epoch": tokens_per_epoch,
        "epochs": cfg.epochs, "global_steps": trainer.state.global_step,
        "selection_steps": checks,
        "resumed_from_step": start_step,
        "train_loss": out.training_loss,
        "dev_loss": [{"step": h["step"], "loss": h["eval_loss"]}
                     for h in trainer.state.log_history if "eval_loss" in h],
        "train_runtime_s_this_session": runtime,
        "tokens_per_s": tokens_per_step * steps_now / runtime if runtime > 0 else None,
        "train_flops": training_flops(base.n_params, tokens_per_epoch * cfg.epochs),
        "training_id": cfg.training_id,
        "lora_targets": LORA_TARGETS, "quantized": base.quantized,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "adapter_dtype": sorted({str(p.dtype).replace("torch.", "")
                                 for p in trainer.model.parameters() if p.requires_grad}),
        "batch": {"batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum,
                  "gradient_checkpointing": cfg.gradient_checkpointing},
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                            if torch.cuda.is_available() else None),
        "log_history": _curve(trainer.state.log_history),
        "versions": library_versions(),
    }
    missing = [s for s in checks if not (home / SELECTION_DIR / f"step-{s}").is_dir()]
    if missing:
        raise RuntimeError(f"selection checkpoints missing for steps {missing}")
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # resume state; the adapters are kept
    log.info("trained: loss %.4f, %d steps, %.0f tok/s", out.training_loss,
             trainer.state.global_step, stats["tokens_per_s"] or 0)
    del trainer, model, base
    _free_memory()
    return stats


def _select_checkpoint(cfg: ExperimentConfig, home: Path) -> dict:
    """Answer the dev slice with every kept checkpoint (same `generate` and
    metrics as the test set) and copy the best to home/adapter. Best = highest
    primary EM, then faithfulness (program arms), then the earlier step."""
    from .faithfulness import program_faithfulness
    from .inference import generate, to_faith_items
    from .metrics import exact_match, primary_exact_match

    kept = sorted((home / SELECTION_DIR).glob("step-*"), key=lambda d: int(d.name[5:]))
    if not kept:
        raise RuntimeError(f"no selection checkpoints in {home / SELECTION_DIR}")
    dev = _dev_examples(cfg)
    t0 = time.time()
    rows: list[dict] = []
    if dev:
        lm = load_student(cfg.model, adapter_dir=kept[0])
        golds = [ex.answer for ex in dev]
        for i, d in enumerate(kept):
            step = int(d.name[5:])
            if i:
                name = d.name.replace("-", "_")
                lm.model.load_adapter(str(d), adapter_name=name)
                lm.model.set_adapter(name)
            preds = generate(lm, cfg, dev)
            answers = [p.answer for p in preds]
            faith = (program_faithfulness(to_faith_items(preds, golds))
                     if any(p.program for p in preds) else None)
            rows.append({"step": step,
                         "dev_em": primary_exact_match(answers, golds,
                                                       [p.percent_form for p in preds]),
                         "dev_em_strict": exact_match(answers, golds),
                         "dev_faithfulness": faith})
            log.info("selection: step %d dev EM %.3f%s", step, rows[-1]["dev_em"],
                     "" if faith is None else f", faithfulness {faith:.3f}")
        del lm
        _free_memory()
        best = max(rows, key=lambda r: (r["dev_em"], r["dev_faithfulness"] or 0.0, -r["step"]))
        chosen = best["step"]
    else:
        chosen = int(kept[-1].name[5:])          # no dev data: the final checkpoint
    dst = home / "adapter"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(home / SELECTION_DIR / f"step-{chosen}", dst)
    log.info("selected checkpoint: step %d (checked %s)", chosen,
             [r["step"] for r in rows] or [chosen])
    return {"checks": rows, "chosen_step": chosen, "dev_examples": len(dev),
            "rule": "max primary dev EM, then dev faithfulness, then earlier step",
            "seconds": round(time.time() - t0, 1)}


def training_dir(cfg: ExperimentConfig, run_dir: str | Path) -> Path:
    """The folder of cfg's adapter: <runs>/_adapters/<training_id>/, shared by
    every experiment (split) with the same training fields."""
    return Path(run_dir).parent / ADAPTERS_DIR / cfg.training_id


def build_sft_records(cfg: ExperimentConfig, examples: list[Example], tokenizer: Any,
                      sample: bool = True) -> tuple[list[dict], int, int]:
    """Conversational prompt/completion records for SFTTrainer.

    Returns (records, n_dropped_too_long, total_tokens). Items longer than
    cfg.max_seq_len are dropped, never truncated (truncation would cut the
    target). If cfg.train_examples is set and `sample` is True, a seeded sample
    of that many items is used.
    """
    sup = cfg.supervision
    if sup == "pot_distilled":
        raise NotImplementedError("A8 trains on teacher traces: load_distilled_traces")
    style = PROMPT_STYLE[sup]
    pool = [ex for ex in examples if sup != "pot_gold" or ex.gold_program]
    if sample and cfg.train_examples:
        pool = random.Random(cfg.seed).sample(pool, min(cfg.train_examples, len(pool)))
    records, dropped, total = [], 0, 0
    for ex in pool:
        prompt = build_messages(ex, style)
        completion = [{"role": "assistant", "content": format_target(ex, sup)}]
        n = _n_tokens(tokenizer, prompt + completion)
        if n > cfg.max_seq_len:
            dropped += 1
            continue
        records.append({"prompt": prompt, "completion": completion})
        total += n
    return records, dropped, total


def load_distilled_traces(cfg: ExperimentConfig) -> list[dict]:
    """Load execution-verified teacher PoT traces for the pot_distilled arm.

    TODO: read the traces produced in Job 3 (or reused from arXiv:2408.12337 if
    released); each item pairs an Example uid with a verified program. Only keep
    programs that executed to the gold answer (verify with ror.sandbox).
    """
    raise NotImplementedError("implement distilled-trace loading")


# --- helpers ---

def _n_tokens(tokenizer: Any, messages: list[dict]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


_CURVE_KEYS = ("step", "epoch", "loss", "learning_rate", "grad_norm", "eval_loss",
               "eval_mean_token_accuracy")


def _curve(log_history: list[dict]) -> list[dict]:
    """The Trainer's log history reduced to the curve fields, as floats."""
    points = []
    for h in log_history:
        point = {}
        for k in _CURVE_KEYS:
            try:
                point[k] = float(h[k])
            except (KeyError, TypeError, ValueError):
                continue
        if len(point) > 2:                # more than step/epoch
            points.append(point)
    return points


def _dev_examples(cfg: ExperimentConfig) -> list[Example]:
    """The fixed dev slice (first cfg.dev_examples items) for checkpoint selection
    and dev loss."""
    try:
        return load_dataset(cfg.dataset, "dev")[:cfg.dev_examples]
    except FileNotFoundError:
        return []


def _checkpoint_step(path: Optional[str]) -> int:
    try:
        return int(Path(path).name.split("-")[-1]) if path else 0
    except ValueError:
        return 0


def _single_device(args: Any) -> Any:
    """Train on the one GPU the student was loaded on.

    With several GPUs visible (Kaggle T4x2) the HF Trainer wraps a model that
    sits on a single device in nn.DataParallel (it exempts 8-bit models, not
    4-bit ones): the replica on cuda:1 crashes, and the effective batch would be
    multiplied by the GPU count. Forcing n_gpu=1 is what the Trainer does itself
    for model-parallel models.
    """
    if args.n_gpu > 1:
        log.info("%d GPUs visible; the student trains on cuda:0 only", args.n_gpu)
        args._n_gpu = 1
    return args


def _adapters_fp32(model: Any) -> int:
    """Keep the trainable (LoRA) weights in fp32; returns how many were cast.

    trl 1.14's SFTTrainer casts the trainable weights of a 4-bit model to bf16.
    The T4 has no bf16, so training runs fp16 autocast with a GradScaler, whose
    CUDA unscale kernel rejects bf16 gradients ("not implemented for
    'BFloat16'"). fp32 adapters (peft's default) work under fp16 and bf16
    autocast alike, so runs stay comparable across GPUs.
    """
    import torch

    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def _check_budget(needed_s: float, why: str) -> None:
    left = seconds_left()
    if left is not None and left < needed_s:
        raise RunPaused(f"{why} ({left / 60:.0f} min left, need {needed_s / 60:.0f})")


def _selection_callback(steps: set, keep_dir: Path, evaluate: bool) -> Any:
    """Trainer callback: save a checkpoint at each selection step, copy its
    adapter files to keep_dir/step-N, and evaluate the dev loss there."""
    from transformers import TrainerCallback

    class SelectionKeeper(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if state.global_step in steps:
                control.should_save = True
                control.should_evaluate = evaluate
            return control

        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            if state.global_step in steps:
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                dst = keep_dir / f"step-{state.global_step}"
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("adapter_*"):
                    shutil.copy2(f, dst / f.name)
            return control

    return SelectionKeeper()


def _time_budget_callback() -> Any:
    """Trainer callback: save and stop once the session deadline is near."""
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        paused = False

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            left = seconds_left()
            if left is not None and left < TRAIN_STOP_RESERVE_S:
                self.paused = True
                control.should_save = True
                control.should_training_stop = True
            return control

    return TimeBudget()


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _link(run_dir: Path, home: Path, cfg: ExperimentConfig, reused: bool) -> None:
    """Record in the experiment's folder which adapter it used."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training.json").write_text(json.dumps({
        "training_id": cfg.training_id, "reused": reused,
        "adapter_dir": (Path("..") / home.relative_to(run_dir.parent)).as_posix(),
    }, indent=2), encoding="utf-8")


def _load_trained(cfg: ExperimentConfig, adapter_dir: Path, stats_path: Path,
                  reused: bool) -> LoadedModel:
    lm = load_student(cfg.model, adapter_dir=adapter_dir)
    lm.train_stats = {**json.loads(stats_path.read_text(encoding="utf-8")),
                      "reused_adapter": reused}
    return lm
