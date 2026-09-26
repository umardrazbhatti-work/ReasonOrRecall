"""Filesystem locations, resolved independently of the current working directory.

Kaggle runs scripts from arbitrary directories, so nothing should depend on the
cwd. The data directory is configurable because on Kaggle the preprocessed data
lives in a read-only attached dataset under /kaggle/input.

    ROR_DATA_DIR   -> directory holding ror_data_manifest.json (default: <repo>/data)
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "ROR_DATA_DIR"
MANIFEST_NAME = "ror_data_manifest.json"


def repo_root() -> Path:
    """The repository root (the directory containing `ror/`)."""
    return Path(__file__).resolve().parents[1]


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    """Root of the data tree: $ROR_DATA_DIR if set, else <repo>/data."""
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env) if env else repo_root() / "data"


def processed_path(dataset: str, split_file: str) -> Path:
    """Path of a preprocessed split, e.g. processed/finqa/test.jsonl."""
    return data_dir() / "processed" / dataset / f"{split_file}.jsonl"


def clean_set_path() -> Path:
    """The contamination-controlled set written by scripts/build_clean_set.py."""
    return data_dir() / "clean_set" / "clean.jsonl"


def control_set_path() -> Path:
    """The style-control set (clean-set templates on FinQA test tables, roadmap
    P2.7), written by scripts/build_clean_set.py next to the clean set."""
    return data_dir() / "clean_set" / "control.jsonl"


def manifest_path() -> Path:
    return data_dir() / MANIFEST_NAME
