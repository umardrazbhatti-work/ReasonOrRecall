"""Experiment configuration: a typed config, a YAML loader that merges
`base.yaml` + per-arm + per-model settings, and a *deterministic* experiment id.

The experiment id is a hash of the identity-defining fields only. Two configs
that would produce the same experiment get the same id, which is what lets the
registry skip work that is already done. Fields that do not change the result
(e.g. logging verbosity) are excluded from the hash on purpose.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from .logging_utils import get_logger

log = get_logger("ror.config")


# Fields that define an experiment's *identity*. Change any of these and it is a
# genuinely different experiment (new id). Keep this list in sync deliberately:
# every field that can change a result belongs here (optimisation settings such
# as batch size / grad accumulation and decoding settings included).
IDENTITY_FIELDS = (
    "arm",
    "model",
    "role",
    "supervision",
    "inference",
    "dataset",
    "split",
    "seed",
    "teacher",
    "train_examples",
    "eval_examples",
    "epochs",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "max_seq_len",
    "batch_size",
    "grad_accum",
    "self_consistency_k",
    "max_new_tokens",
    "temperature",
    "phase",
)

# Fields that define a *trained adapter*: no split, no evaluation or decoding
# settings. Experiments that differ only in those (the standard, clean and
# control splits) share one adapter, so each model is trained once and
# evaluated on every split (roadmap P1.1).
TRAINING_FIELDS = (
    "arm",
    "model",
    "supervision",
    "dataset",
    "seed",
    "teacher",
    "train_examples",
    "epochs",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "max_seq_len",
    "batch_size",
    "grad_accum",
)


@dataclass
class ExperimentConfig:
    # --- identity ---
    arm: str                      # A1..A8
    model: str                    # key into configs/models.yaml
    role: str = "student"         # "student" | "large"
    supervision: str = "none"     # none | answer | cot | pot_gold | pot_distilled
    inference: str = "greedy"     # zeroshot | fewshot | greedy | self_consistency
    dataset: str = "finqa"        # finqa | convfinqa | tatqa
    split: str = "standard"       # standard | clean
    seed: int = 0
    teacher: Optional[str] = None  # model key used to generate distilled traces
    phase: int = 1

    # --- training hyperparameters (identity-affecting) ---
    train_examples: Optional[int] = None  # None = all (else a seeded sample)
    eval_examples: Optional[int] = None   # None = all (else the first N; smoke runs)
    epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 16

    # --- inference (identity-affecting) ---
    self_consistency_k: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0

    # --- non-identity / operational (excluded from the id hash) ---
    infer_batch_size: int = 8      # generation batch size
    name: str = ""                 # human label; derived if empty
    max_attempts: int = 2
    notes: str = ""

    def identity_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in IDENTITY_FIELDS}

    @property
    def experiment_id(self) -> str:
        """Stable 12-char hash of the identity fields."""
        blob = json.dumps(self.identity_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def training_id(self) -> str:
        """Stable 12-char hash of TRAINING_FIELDS: the id of the adapter this
        experiment trains or reuses."""
        d = asdict(self)
        blob = json.dumps({k: d[k] for k in TRAINING_FIELDS}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def default_name(self) -> str:
        parts = [self.arm, self.model, self.dataset, self.split, f"s{self.seed}"]
        if self.self_consistency_k > 1:
            parts.append(f"k{self.self_consistency_k}")
        return "-".join(parts)

    def resolved_name(self) -> str:
        return self.name or self.default_name()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Cast numeric strings to the field's type. PyYAML reads `2e-4` (no dot) as
    a *string*, which would silently change the experiment id and break training."""
    if not isinstance(value, str) or isinstance(default, bool):
        return value
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    return value


def config_from_dict(d: dict) -> ExperimentConfig:
    """Build an ExperimentConfig from a dict.

    Unknown keys are ignored (so config files can carry annotations) but logged,
    because a typo such as `learning_rte` would otherwise fall back to the default
    without anyone noticing. Numeric strings are coerced to the field's type.
    """
    fields = ExperimentConfig.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = sorted(k for k in d if k not in fields)
    if unknown:
        log.warning("ignoring unknown config keys: %s", ", ".join(unknown))
    kwargs = {k: _coerce(k, v, fields[k].default) for k, v in d.items() if k in fields}
    return ExperimentConfig(**kwargs)


def load_experiment_config(
    experiment_yaml: str | Path,
    base_yaml: str | Path | None = None,
) -> ExperimentConfig:
    """Merge base defaults with an experiment file into one ExperimentConfig."""
    base = load_yaml(base_yaml) if base_yaml else {}
    exp = load_yaml(experiment_yaml)
    merged = _deep_merge(base.get("defaults", base), exp)
    return config_from_dict(merged)
