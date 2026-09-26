"""Model loading — QLoRA students and the large teacher/baseline.

IMPLEMENT the functions that raise NotImplementedError. Model keys resolve via
configs/models.yaml (name, family, size_b, hf_id, training cutoff, tier).

Guardrail (proposal 2, §6): a ≥70B model is Phase 2 only and must NOT be loaded
on free hardware. `load_teacher` must refuse a ≥70B model when phase == 1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import load_yaml
from .logging_utils import get_logger
from .paths import repo_root

log = get_logger("ror.models")

# The guardrail reads size_b / phase2_only from configs/models.yaml.
FREE_TIER_MAX_SIZE_B = 34.0  # ~32B fits on T4x2 in 4-bit; 70B does not

# "cpu" forces an unquantized fp32 CPU load (tests, local checks)
DEVICE_ENV = "ROR_DEVICE"

# peft refuses torchao older than this (ImportError when attaching LoRA layers)
PEFT_MIN_TORCHAO = "0.16.0"


def hide_incompatible_torchao() -> bool:
    """Hide an installed-but-incompatible torchao from this process.

    The Kaggle image ships torchao 0.10 (for torchtune). peft 0.19 raises
    ImportError on any torchao older than 0.16 when it attaches LoRA layers,
    instead of treating it as absent. This project never uses torchao, so an
    incompatible install is blocked with `sys.modules["torchao"] = None` (Python's
    documented way to make an import fail); `importlib.util.find_spec` then
    reports it as missing and peft skips its torchao backend. A compatible or
    already-imported torchao is left alone. Returns True if it hid torchao.
    """
    import sys
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    if "torchao" in sys.modules:
        return False
    try:
        installed = version("torchao")
    except PackageNotFoundError:
        return False
    if Version(installed) >= Version(PEFT_MIN_TORCHAO):
        return False
    sys.modules["torchao"] = None  # type: ignore[assignment]
    log.warning("hiding torchao %s: peft needs >= %s and this project does not use it",
                installed, PEFT_MIN_TORCHAO)
    return True


@dataclass
class LoadedModel:
    """A model ready for generation, plus what the experiment needs to log."""
    model: Any
    tokenizer: Any
    spec: dict                      # configs/models.yaml entry
    quantized: bool                 # 4-bit NF4 (GPU) vs full precision (CPU)
    adapter_dir: Optional[str] = None
    train_stats: dict = field(default_factory=dict)  # filled by ror.training

    @property
    def n_params(self) -> float:
        return float(self.spec.get("size_b") or 0.0) * 1e9


def use_cuda() -> bool:
    """True if a CUDA GPU is available and not overridden by ROR_DEVICE=cpu."""
    import torch

    return os.environ.get(DEVICE_ENV, "").lower() != "cpu" and torch.cuda.is_available()


def gpus_used(model: Any) -> int:
    """How many GPUs a loaded model sits on (for the cost of a run); 0 on CPU."""
    if not use_cuda():
        return 0
    device_map = None
    for m in (model, getattr(model, "base_model", None),
              getattr(getattr(model, "base_model", None), "model", None)):
        device_map = getattr(m, "hf_device_map", None) if m is not None else None
        if device_map:
            break
    devices = {d for d in (device_map or {}).values() if d not in ("cpu", "disk")}
    return max(1, len(devices))


def compute_dtype() -> Any:
    """fp16 on pre-Ampere GPUs (the T4 has no bf16), bf16 on Ampere+, fp32 on CPU.
    (torch.cuda.is_bf16_supported() also reports emulated bf16, so check the
    compute capability instead.)"""
    import torch

    if not use_cuda():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_tokenizer(hf_id: str) -> Any:
    """Tokenizer with a pad token and left padding (for batched generation)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_student(name: str, adapter_dir: Optional[str | Path] = None) -> LoadedModel:
    """Load a ≤8B student: 4-bit NF4 (double quantization, fp16/bf16 compute) on
    a single GPU, or full precision on CPU. If `adapter_dir` is given the trained
    LoRA adapter is attached; otherwise the bare base model is returned and
    ror.training attaches a fresh LoRA with the experiment's r/alpha/dropout.
    """
    hide_incompatible_torchao()   # before transformers/peft probe for it
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    spec = resolve_model(name)
    cuda = use_cuda()
    dtype = compute_dtype()
    kwargs: dict[str, Any] = {"dtype": dtype}
    if cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}   # one GPU; QLoRA ≤8B fits on a T4
    else:
        log.warning("no CUDA GPU (or ROR_DEVICE=cpu): loading %s unquantized on CPU",
                    spec["hf_id"])
    log.info("loading %s (%s)%s", name, spec["hf_id"],
             f" + adapter {adapter_dir}" if adapter_dir else "")
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return LoadedModel(model=model, tokenizer=load_tokenizer(spec["hf_id"]), spec=spec,
                       quantized=cuda, adapter_dir=str(adapter_dir) if adapter_dir else None)


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
