"""QLoRA supervised fine-tuning (Job 1).

`train_qlora` is **resumable**: checkpoints go to runs/<exp_id>/checkpoints every
SAVE_STEPS optimizer steps and training resumes from the latest one. It also
respects the session time budget ($ROR_DEADLINE_UNIX, set by the Kaggle
notebook): shortly before the deadline it saves, stops, and raises RunPaused so
the next session continues instead of being hard-killed mid-step.

Outputs in the run directory:
  adapter/          the trained LoRA adapter (the product; checkpoints are
                    deleted once it is saved)
  train_stats.json  counts, losses, throughput, estimated FLOPs, versions
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
DEV_EVAL_EXAMPLES = 200           # fixed dev slice for per-epoch eval loss
TRAIN_STOP_RESERVE_S = 45 * 60    # stop training this long before the deadline
MIN_INFERENCE_S = 20 * 60         # after training, pause if less time than this is left


def train_qlora(
    cfg: ExperimentConfig,
    examples_train: list[Example],
    run_dir: str | Path,
) -> LoadedModel:
    """Fine-tune a student with QLoRA under cfg.supervision and return it, loaded
    with the trained adapter and ready for `ror.inference.generate`.

    Reuses a finished adapter in run_dir if one exists (e.g. training finished
    but a later step failed), resumes from the latest checkpoint otherwise.
    """
    run_dir = Path(run_dir)
    adapter_dir = run_dir / "adapter"
    stats_path = run_dir / "train_stats.json"
    if (adapter_dir / "adapter_config.json").exists() and stats_path.exists():
        log.info("adapter already trained, reusing %s", adapter_dir)
        return _load_trained(cfg, adapter_dir, stats_path)
    _check_budget(TRAIN_STOP_RESERVE_S, "not enough session time left to train")

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
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))

    dtype = compute_dtype()
    ckpt_dir = run_dir / "checkpoints"
    steps_per_epoch = math.ceil(len(train_recs) / (cfg.batch_size * cfg.grad_accum))
    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(WARMUP_FRACTION * steps_per_epoch * cfg.epochs),
        logging_steps=10,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="epoch" if dev_recs else "no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tok,
        train_dataset=Dataset.from_list(train_recs),
        eval_dataset=Dataset.from_list(dev_recs) if dev_recs else None,
        callbacks=[budget],
    )
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    start_step = _checkpoint_step(last)
    log.info("QLoRA: %s", f"resuming from {last}" if last else "starting fresh")

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
        "resumed_from_step": start_step,
        "train_loss": out.training_loss,
        "eval_loss_by_epoch": [h["eval_loss"] for h in trainer.state.log_history
                               if "eval_loss" in h],
        "train_runtime_s_this_session": runtime,
        "tokens_per_s": tokens_per_step * steps_now / runtime if runtime > 0 else None,
        "train_flops": training_flops(base.n_params, tokens_per_epoch * cfg.epochs),
        "lora_targets": LORA_TARGETS, "quantized": base.quantized,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "versions": library_versions(),
    }
    trainer.save_model(str(adapter_dir))
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # the adapter is the product
    log.info("trained: loss %.4f, %d steps, %.0f tok/s", out.training_loss,
             trainer.state.global_step, stats["tokens_per_s"] or 0)

    del trainer, model, base
    _free_memory()
    _check_budget(MIN_INFERENCE_S, "trained; not enough session time left for inference")
    return _load_trained(cfg, adapter_dir, stats_path)


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


def _dev_examples(cfg: ExperimentConfig) -> list[Example]:
    """A fixed dev slice for eval loss (smaller for smoke runs)."""
    n = DEV_EVAL_EXAMPLES if not cfg.train_examples else max(8, cfg.train_examples // 4)
    try:
        return load_dataset(cfg.dataset, "dev")[:n]
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


def _check_budget(needed_s: float, why: str) -> None:
    left = seconds_left()
    if left is not None and left < needed_s:
        raise RunPaused(f"{why} ({left / 60:.0f} min left, need {needed_s / 60:.0f})")


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


def _load_trained(cfg: ExperimentConfig, adapter_dir: Path, stats_path: Path) -> LoadedModel:
    lm = load_student(cfg.model, adapter_dir=adapter_dir)
    lm.train_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return lm
