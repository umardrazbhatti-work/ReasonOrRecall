"""GPU-hour estimates for suite plans (roadmap §5, P1.7).

`estimate_plan` prices a list of experiments the way they will actually run:
an adapter (training_id) is trained once and only if it does not exist yet;
every experiment then costs its own evaluation. Speeds come from
configs/throughput.yaml (3B measured, the rest scaled until their pilots run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .config import ExperimentConfig, load_yaml
from .data import PROMPT_STYLE
from .paths import configs_dir

TRAINED_ARMS = {"A5", "A6", "A7", "A8"}


def load_throughput(path: Optional[str | Path] = None) -> dict:
    return load_yaml(Path(path) if path else configs_dir() / "throughput.yaml")


def _answers_per_s(cfg: ExperimentConfig, thr: dict) -> float:
    style = PROMPT_STYLE.get(cfg.supervision, "answer")
    speeds = (thr.get("answers_per_s") or {}).get(cfg.model) or {}
    rate = float(speeds.get(style) or speeds.get("answer") or 0.1)
    if cfg.inference == "fewshot":
        rate *= float(thr.get("fewshot_speed_factor", 0.35))
    return rate


def train_hours(cfg: ExperimentConfig, thr: dict) -> float:
    """Training plus checkpoint selection for cfg's adapter."""
    per_q = float((thr.get("tokens_per_question") or {}).get(cfg.dataset, 1100))
    n = cfg.train_examples or int((thr.get("train_questions") or {}).get(cfg.dataset, 6000))
    tok_s = float((thr.get("train_tokens_per_s") or {}).get(cfg.model) or 100.0)
    training = n * per_q * cfg.epochs / tok_s / 3600
    selection = cfg.selection_checks * cfg.dev_examples / _answers_per_s(cfg, thr) / 3600
    return training + selection + float(thr.get("model_load_hours", 0.03))


def eval_hours(cfg: ExperimentConfig, thr: dict) -> float:
    """Answering the evaluation split (plus loading the model)."""
    n = cfg.eval_examples or int((thr.get("eval_items") or {}).get(cfg.split, 1000))
    return n / _answers_per_s(cfg, thr) / 3600 + float(thr.get("model_load_hours", 0.03))


def estimate_plan(configs: Iterable[ExperimentConfig], trained: set[str] = frozenset(),
                  thr: Optional[dict] = None) -> dict:
    """{"total", "train", "eval", "adapters", "evaluations"} in Kaggle session
    hours (what the weekly quota spends; the ~32B model uses both T4s of one
    session) for running `configs`; `trained` holds training_ids whose adapter
    already exists."""
    thr = thr if thr is not None else load_throughput()
    seen = set(trained)
    train_h = eval_h = 0.0
    adapters = evaluations = 0
    for cfg in configs:
        if cfg.arm in TRAINED_ARMS and cfg.training_id not in seen:
            seen.add(cfg.training_id)
            train_h += train_hours(cfg, thr)
            adapters += 1
        eval_h += eval_hours(cfg, thr)
        evaluations += 1
    return {"total": round(train_h + eval_h, 1), "train": round(train_h, 1),
            "eval": round(eval_h, 1), "adapters": adapters, "evaluations": evaluations}
