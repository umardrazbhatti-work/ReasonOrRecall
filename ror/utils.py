"""Small shared helpers: deterministic seeding, git commit capture, and simple
FLOP estimates used to place runs on the accuracy-per-compute frontier.

The FLOP figures are the standard first-order approximations, good enough for a
*relative* frontier across arms/sizes. They are documented, not exact.
"""
from __future__ import annotations

import os
import random
import subprocess
from typing import Optional


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit(short: bool = True) -> str:
    """Current git commit, or '' if not a repo."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


# --- FLOP estimates (first-order; see docstring) ---

def training_flops(n_params: float, n_tokens: float, lora: bool = True) -> float:
    """Approx training FLOPs.

    Full fine-tuning ~ 6 * N * T (fwd 2ND + bwd 4ND). QLoRA skips most of the
    weight-gradient work, so we use ~4 * N * T. Gradient checkpointing recomputes
    the forward pass, adding ~1/3, captured by the 1.33 factor.
    """
    coef = 4.0 if lora else 6.0
    return coef * n_params * n_tokens * 1.33


def inference_flops(n_params: float, prompt_tokens: float, gen_tokens: float,
                    n_examples: int, samples: int = 1) -> float:
    """Approx inference FLOPs ~ 2 * N * tokens. Prefill counts the prompt once;
    decoding counts generated tokens per sample."""
    prefill = 2.0 * n_params * prompt_tokens * n_examples
    decode = 2.0 * n_params * gen_tokens * n_examples * samples
    return prefill + decode
