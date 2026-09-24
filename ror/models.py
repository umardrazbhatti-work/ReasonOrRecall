"""Model loading — QLoRA students and the large teacher/baseline.

IMPLEMENT the functions that raise NotImplementedError. Model keys resolve via
configs/models.yaml (name, family, size_b, hf_id, training cutoff, tier).

Guardrail (proposal 2, §6): a ≥70B model is Phase 2 only and must NOT be loaded
on free hardware. `load_teacher` must refuse a ≥70B model when phase == 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Loaded from configs/models.yaml at implementation time; the guardrail below
# reads size_b from there.
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

    TODO + GUARDRAIL: resolve size_b from configs/models.yaml and raise
    ValueError if phase == 1 and size_b > FREE_TIER_MAX_SIZE_B — a 70B model does
    not fit on free hardware and must wait for Phase 2 (Job 4).
    """
    raise NotImplementedError("implement teacher loading + the ≥70B phase-1 guard")


def resolve_model(name: str, models_yaml: str | Path = "configs/models.yaml") -> dict:
    """Return the model registry entry (hf_id, family, size_b, cutoff, tier).

    TODO: read configs/models.yaml and return the entry for `name`.
    """
    raise NotImplementedError("implement model registry lookup")
