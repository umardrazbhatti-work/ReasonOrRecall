"""Kaggle runtime helpers — the logic behind notebooks/kaggle_runner.ipynb.

The notebook is a thin entry point; everything it does beyond cloning the repo
lives here so it is versioned and tested:

  - load_secrets      Kaggle notebook secrets (HF_TOKEN, ...) -> env vars
  - ensure_data_dir   find the attached ror-data dataset (or extract its zip)
  - restore_runs      carry the experiment registry + results + checkpoints over
                      from a previous notebook version's output, so finished
                      experiments are never repeated and killed runs resume
  - gpu_summary       what accelerator this session got
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .paths import MANIFEST_NAME
from .registry import Status

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
RESULTS_FILE = "results.jsonl"


def on_kaggle() -> bool:
    return KAGGLE_INPUT.exists()


def load_secrets(names: Iterable[str] = ("HF_TOKEN",)) -> list[str]:
    """Copy Kaggle notebook secrets into environment variables of the same name.

    Returns the names that are now set — never the values. Names already present
    in the environment are kept. Outside Kaggle this only reports what is set.
    """
    names = list(names)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except ImportError:
        return [n for n in names if os.environ.get(n)]
    client = UserSecretsClient()
    found = []
    for n in names:
        if not os.environ.get(n):
            try:
                value = client.get_secret(n)
            except Exception:  # noqa: BLE001 — secret not attached to this notebook
                value = None
            if value:
                os.environ[n] = value
        if os.environ.get(n):
            found.append(n)
    return found


def _walk_dirs(roots: Iterable[Path], max_depth: int) -> Iterator[Path]:
    """Directories under `roots`, breadth-first, at most `max_depth` levels deep
    (attached model inputs can be huge; never scan them exhaustively)."""
    frontier = [(Path(r), 0) for r in roots if Path(r).is_dir()]
    while frontier:
        d, depth = frontier.pop(0)
        yield d
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            continue
        frontier += [(c, depth + 1) for c in children]


def find_data_dir(roots: Iterable[Path] = (KAGGLE_INPUT,), max_depth: int = 6) -> Optional[Path]:
    """The directory holding ror_data_manifest.json, or None."""
    for d in _walk_dirs(roots, max_depth):
        if (d / MANIFEST_NAME).is_file():
            return d
    return None


def ensure_data_dir(roots: Iterable[Path] = (KAGGLE_INPUT,),
                    extract_to: Path = Path("/tmp/ror-data"),
                    max_depth: int = 6) -> Optional[Path]:
    """Find the attached dataset. Kaggle normally unpacks an uploaded zip; if it
    kept `ror-data*.zip` as a file instead, extract it to `extract_to`."""
    roots = [Path(r) for r in roots]
    found = find_data_dir(roots, max_depth)
    if found:
        return found
    for d in _walk_dirs(roots, max_depth):
        for z in sorted(d.glob("ror-data*.zip")):
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(extract_to)
            return find_data_dir([extract_to], max_depth)
    return None


def runs_dir_for(suite_path: str | Path, working: Path = KAGGLE_WORKING) -> Path:
    """Where a suite's registry lives on Kaggle: /kaggle/working/<suite runs_dir>.
    Keeps e.g. smoke runs (runs_smoke) apart from the study (runs)."""
    from .config import load_yaml

    return Path(working) / Path(load_yaml(suite_path).get("runs_dir", "runs")).name


def latest_results(runs_dir: Path, n: int = 5) -> list[str]:
    """One human-readable line per most recent result row."""
    from .results import load_results

    lines = []
    for r in load_results(runs_dir)[-n:]:
        train = (r.get("extra") or {}).get("train") or {}
        parts = [f"{r['name']}: EM={_pct(r.get('exact_match'))} on {r.get('n_examples')} items",
                 f"wall {(r.get('wall_time_s') or 0) / 60:.1f} min"]
        if train:
            parts.append(f"trained on {train.get('n_train')} items, "
                         f"loss {train.get('train_loss') or float('nan'):.4f}, "
                         f"{train.get('tokens_per_s') or 0:.0f} tok/s")
        parts.append(str(r.get("gpu", "")))
        lines.append(" | ".join(parts))
    return lines


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _runs_sources(roots: Iterable[Path], dest: Path, max_depth: int) -> list[Path]:
    """Previous runs directories with the same name as `dest` (so a smoke registry
    never mixes with the study's) that hold results.jsonl or */status.json,
    newest first."""
    dest = dest.resolve()
    found = []
    for d in _walk_dirs(roots, max_depth):
        if d.resolve() == dest or d.name != dest.name:
            continue
        if (d / RESULTS_FILE).is_file() or any(d.glob("*/status.json")):
            files = [p for p in d.rglob("*") if p.is_file()]
            newest = max((p.stat().st_mtime for p in files), default=0.0)
            found.append((newest, d))
    return [d for _, d in sorted(found, key=lambda x: -x[0])]


def restore_runs(dest: Path, roots: Iterable[Path] = (KAGGLE_INPUT,),
                 max_depth: int = 6) -> dict:
    """Restore the registry, results and checkpoints from previous outputs.

    Attach an earlier version of this notebook as an input and its `runs/` is
    merged into `dest`:
      - files already in `dest` are never overwritten (newest source wins);
      - results.jsonl lines are merged without duplicates;
      - a restored status "running" means that session was killed (Kaggle runs
        one session per notebook), so it becomes "failed" — the attempt was
        already counted — and the registry retries it within max_attempts,
        resuming training from its last checkpoint.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    report = {"sources": [], "files_copied": 0, "results_added": 0, "interrupted": []}
    for src in _runs_sources(roots, dest, max_depth):
        report["sources"].append(str(src))
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            target = dest / rel
            if rel.as_posix() == RESULTS_FILE:
                report["results_added"] += _merge_results(p, target)
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            report["files_copied"] += 1
            if p.name == "status.json" and _mark_interrupted(target):
                report["interrupted"].append(rel.parent.as_posix())
    return report


def _merge_results(src: Path, target: Path) -> int:
    have = set(target.read_text(encoding="utf-8").splitlines()) if target.exists() else set()
    new = [ln for ln in src.read_text(encoding="utf-8").splitlines()
           if ln.strip() and ln not in have]
    if new:
        with open(target, "a", encoding="utf-8") as f:
            f.write("".join(ln + "\n" for ln in new))
    return len(new)


def _mark_interrupted(status_file: Path) -> bool:
    try:
        st = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if st.get("status") != Status.RUNNING.value:
        return False
    st["status"] = Status.FAILED.value
    st["last_error"] = "interrupted: the Kaggle session ended while running (restored)"
    status_file.write_text(json.dumps(st, indent=2, sort_keys=True), encoding="utf-8")
    return True


def gpu_summary() -> str:
    """One line describing the accelerator, with a warning for P100 (no NF4 kernels)."""
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU — set Accelerator to 'GPU T4 x2' in the notebook settings"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    line = f"{len(names)} GPU(s): {', '.join(names)}"
    if any("P100" in n for n in names):
        line += "  WARNING: P100 lacks the 4-bit NF4 kernels QLoRA needs — use T4 x2"
    return line
