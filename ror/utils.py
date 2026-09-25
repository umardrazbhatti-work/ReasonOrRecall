"""Small shared helpers: deterministic seeding, git commit capture, and simple
FLOP estimates used to place runs on the accuracy-per-compute frontier.

The FLOP figures are the standard first-order approximations, good enough for a
*relative* frontier across arms/sizes. They are documented, not exact.
"""
from __future__ import annotations

import os
import random
import subprocess
import time
from pathlib import Path
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
    """Commit of the repo this package lives in, or '' if it is not a git checkout.

    Runs git in the repository root (not the cwd, which on Kaggle is elsewhere)
    and appends '-dirty' when tracked files have uncommitted changes, so a result
    is never attributed to a commit that does not contain the code that made it.
    """
    root = str(Path(__file__).resolve().parents[1])
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root,
                               stderr=subprocess.DEVNULL).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return ""


def library_versions(names: tuple[str, ...] = ("torch", "transformers", "trl", "peft",
                                               "bitsandbytes", "accelerate", "datasets")
                     ) -> dict[str, Optional[str]]:
    """Installed versions of the ML stack, recorded with every result."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


# --- session time budget (Kaggle sessions are hard-killed at ~12h) ---

DEADLINE_ENV = "ROR_DEADLINE_UNIX"


class RunPaused(Exception):
    """Raised when a run stops on purpose (session time budget) with its progress
    saved. The runner returns it to PENDING without consuming an attempt; the next
    session resumes it from the checkpoint."""


def seconds_left() -> Optional[float]:
    """Seconds until the session deadline ($ROR_DEADLINE_UNIX), or None if unset."""
    raw = os.environ.get(DEADLINE_ENV)
    if not raw:
        return None
    try:
        return float(raw) - time.time()
    except ValueError:
        return None


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
