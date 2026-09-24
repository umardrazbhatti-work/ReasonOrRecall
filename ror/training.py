"""QLoRA supervised fine-tuning (Job 1).

IMPLEMENT `train_qlora`. It must be **resumable** — a killed Kaggle session must
not lose a run. Checkpoint adapters to `run_dir` and resume from the latest.

Keep the signature; `ror.experiment._execute` calls it and expects a model
object usable by `ror.inference.generate`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import Example, format_target


def train_qlora(
    cfg: ExperimentConfig,
    examples_train: list[Example],
    run_dir: str | Path,
) -> Any:
    """Fine-tune a student with QLoRA under cfg.supervision and return the model.

    TODO:
      - build targets with `format_target` (or, for pot_distilled, load verified
        teacher traces produced in Job 3)
      - load base + LoRA via ror.models.load_student
      - trl SFTTrainer with cfg hyperparameters (lr, epochs, r/alpha, batch,
        grad_accum, max_seq_len); gradient checkpointing ON
      - save adapters + trainer state to run_dir; resume if a checkpoint exists
      - record est. training FLOPs via ror.utils.training_flops into the result
        (the caller reads it if you attach it to the returned object or cfg)

    Start with ONE arm end-to-end (A5, Qwen2.5-3B, FinQA) before scaling.
    """
    raise NotImplementedError("implement QLoRA SFT — see docstring and proposal 5.4")


def load_distilled_traces(cfg: ExperimentConfig) -> list[dict]:
    """Load execution-verified teacher PoT traces for the pot_distilled arm.

    TODO: read the traces produced in Job 3 (or reused from arXiv:2408.12337 if
    released); each item pairs an Example uid with a verified program. Only keep
    programs that executed to the gold answer (verify with ror.sandbox).
    """
    raise NotImplementedError("implement distilled-trace loading")
