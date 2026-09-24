"""Model loading — QLoRA students and the large teacher/baseline.

IMPLEMENT the functions that raise NotImplementedError. Model keys resolve via
configs/models.yaml (name, family, size_b, hf_id, training cutoff, tier).

Guardrail (proposal 2, §6): a ≥70B model is Phase 2 only and must NOT be loaded
on free hardware. `load_teacher` must refuse a ≥70B model when phase == 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config import load_yaml
from .paths import repo_root

# The guardrail reads size_b / phase2_only from configs/models.yaml.
FREE_TIER_MAX_SIZE_B = 34.0  # ~32B fits on T4x2 in 4-bit; 70B does not


def load_student(name: str, adapter_dir: Optional[str | Path] = None) -> Any:
    """Load a ≤8B student in 4-bit (NF4) with a LoRA adapter attached.

    TODO:
      - resolve `name` via configs/models.yaml -> hf_id, size_b
      - load base with bitsandbytes 4-bit (nf4, double-quant, bf16 compute)
      - attach a fresh peft LoraConfig, or load `adapter_dir` if given
      - return an object exposing generation (the wrapper `ror.inference` expects)
    """
    raise NotImplementedError("implement 4-bit QLoRA student loading")


def load_teacher(name: str, phase: int = 1) -> Any:
    """Load the large baseline / distillation teacher.

    Phase 1: a ~32B open model on the dual-T4 (device_map='auto', 4-bit).
    Phase 2: a ≥70B model via vLLM on a rented GPU, or an API client.

    GUARDRAIL (implemented): `check_phase_allowed` raises ValueError before any
    weights are touched if a ≥70B / phase2_only model is requested in phase 1.
    TODO: the actual loading (Job 3).
    """
    check_phase_allowed(name, phase)
    raise NotImplementedError("implement teacher loading (Job 3)")


def check_phase_allowed(name: str, phase: int) -> None:
    """Raise ValueError if `name` must not be loaded in `phase`.

    A ≥70B model needs ~40 GB in 4-bit and does not fit on free hardware
    (Kaggle T4x2 = 32 GB), so it is Phase 2 (Job 4) only.
    """
    spec = resolve_model(name)
    size_b = float(spec.get("size_b") or 0.0)
    if phase == 1 and (spec.get("phase2_only") or size_b > FREE_TIER_MAX_SIZE_B):
        raise ValueError(
            f"{name} ({size_b:g}B) is Phase 2 only: it does not fit on free hardware "
            f"(limit {FREE_TIER_MAX_SIZE_B:g}B in phase 1). Run it in Job 4 on paid compute."
        )


def resolve_model(name: str, models_yaml: str | Path = "configs/models.yaml") -> dict:
    """Return the model registry entry (hf_id, family, size_b, cutoff, tier), plus
    its key under `name`. A relative `models_yaml` is resolved against the repo
    root, so this works from any working directory."""
    path = Path(models_yaml)
    if not path.is_absolute():
        path = repo_root() / path
    models = load_yaml(path).get("models", {})
    if name not in models:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(models))}")
    return {"name": name, **models[name]}
