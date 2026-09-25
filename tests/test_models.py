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


def test_incompatible_torchao_is_hidden_from_peft(monkeypatch):
    import importlib.metadata as md
    import importlib.util

    import ror.models as models

    real_version = md.version
    monkeypatch.setattr(md, "version",
                        lambda name: "0.10.0" if name == "torchao" else real_version(name))
    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    try:
        assert models.hide_incompatible_torchao() is True
        assert importlib.util.find_spec("torchao") is None   # what peft/transformers probe
        with pytest.raises(ImportError):
            import torchao  # noqa: F401
    finally:
        sys.modules.pop("torchao", None)


def test_compatible_or_absent_torchao_is_left_alone(monkeypatch):
    import importlib.metadata as md

    import ror.models as models

    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    monkeypatch.setattr(md, "version", lambda name: "0.16.0")
    assert models.hide_incompatible_torchao() is False and "torchao" not in sys.modules

    def missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", missing)
    assert models.hide_incompatible_torchao() is False


def test_peft_can_attach_lora_in_this_environment():
    # Kaggle ships torchao 0.10; without the guard peft raises ImportError here.
    iu = pytest.importorskip("peft.import_utils")
    import ror.models as models

    models.hide_incompatible_torchao()
    iu.is_torchao_available()  # must not raise
