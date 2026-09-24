"""Tests for the model registry lookup and the ≥70B phase-1 guardrail."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.models import check_phase_allowed, load_teacher, resolve_model


def test_resolve_model_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # configs/ must be found relative to the repo, not cwd
    spec = resolve_model("qwen2.5-3b")
    assert spec["hf_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["tier"] == "student" and spec["name"] == "qwen2.5-3b"


def test_resolve_unknown_model():
    with pytest.raises(KeyError):
        resolve_model("no-such-model")


def test_70b_refused_in_phase1():
    with pytest.raises(ValueError, match="Phase 2"):
        check_phase_allowed("qwen2.5-72b", phase=1)
    with pytest.raises(ValueError, match="Phase 2"):
        load_teacher("qwen2.5-72b", phase=1)  # guard fires before any loading


def test_32b_allowed_in_phase1_and_70b_in_phase2():
    check_phase_allowed("qwen2.5-32b", phase=1)
    check_phase_allowed("qwen2.5-72b", phase=2)
