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


# Fields that define an experiment's *identity*. Change any of these and it is a
# genuinely different experiment (new id). Keep this list in sync deliberately.
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
    "epochs",
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "max_seq_len",
    "self_consistency_k",
    "phase",
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
    train_examples: Optional[int] = None  # None = all
    epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 16

    # --- inference (identity-affecting where noted) ---
    self_consistency_k: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0

    # --- non-identity / operational (excluded from the id hash) ---
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
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def config_from_dict(d: dict) -> ExperimentConfig:
    """Build an ExperimentConfig from a dict, ignoring unknown keys (so config
    files can carry extra annotations without breaking construction)."""
    known = {f for f in ExperimentConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    return ExperimentConfig(**{k: v for k, v in d.items() if k in known})


def load_experiment_config(
    experiment_yaml: str | Path,
    base_yaml: str | Path | None = None,
) -> ExperimentConfig:
    """Merge base defaults with an experiment file into one ExperimentConfig."""
    base = load_yaml(base_yaml) if base_yaml else {}
    exp = load_yaml(experiment_yaml)
    merged = _deep_merge(base.get("defaults", base), exp)
    return config_from_dict(merged)
