"""End-to-end A5 through the real runner with a tiny model on CPU.

Exercises the actual trl/peft/transformers code paths (SFT with completion-only
loss, LoRA, checkpoint, pause + resume, generation, scoring, logging) against
whatever library versions are installed — on Kaggle, the pinned ones. The 4-bit
GPU path is covered by the smoke suite. Skipped if the ML stack or the tiny
model (a few MB from the Hugging Face Hub) is unavailable.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.config import ExperimentConfig  # noqa: E402
from ror.registry import Registry, Status  # noqa: E402
from ror.results import load_results  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def tiny_model(monkeypatch, fake_data_dir):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # noqa: BLE001 — offline, hub down, ...
        pytest.skip(f"tiny model unavailable: {e}")
    import ror.models as models

    spec = {"name": "qwen2.5-3b", "hf_id": TINY, "family": "qwen2.5",
            "size_b": 0.0024, "tier": "student"}
    monkeypatch.setattr(models, "resolve_model", lambda name, *a, **k: dict(spec))
    monkeypatch.setenv("ROR_DEVICE", "cpu")
    monkeypatch.delenv("ROR_DEADLINE_UNIX", raising=False)
    return spec


def _cfg(**kw) -> ExperimentConfig:
    base = dict(arm="A5", model="qwen2.5-3b", supervision="answer", epochs=2,
                batch_size=1, grad_accum=1, max_new_tokens=8, infer_batch_size=2)
    return ExperimentConfig(**{**base, **kw})


def test_a5_end_to_end_logs_a_result(tmp_path, tiny_model):
    from ror.experiment import run_experiment

    runs = tmp_path / "runs"
    cfg = _cfg()
    res = run_experiment(cfg, runs_dir=runs)
    assert res is not None, (runs / cfg.experiment_id / "error.txt").read_text() \
        if (runs / cfg.experiment_id / "error.txt").exists() else "no result"
    assert 0.0 <= res.exact_match <= 1.0 and res.n_examples == 2
    assert res.train_flops > 0 and res.infer_flops > 0
    assert res.extra["train"]["n_train"] == 2 and res.extra["train"]["global_steps"] == 4

    run_dir = runs / cfg.experiment_id
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    assert not (run_dir / "checkpoints").exists()        # removed once the adapter exists
    rows = [json.loads(line) for line in
            (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {"uid", "gold", "pred", "correct", "text"} <= set(rows[0])

    logged = load_results(runs)
    assert len(logged) == 1 and logged[0]["config"]["max_seq_len"] == cfg.max_seq_len
    assert Registry(runs).load(cfg.experiment_id).status == Status.COMPLETED.value
    assert run_experiment(cfg, runs_dir=runs) is None      # never repeated


def test_pause_mid_training_then_resume_from_checkpoint(tmp_path, tiny_model, monkeypatch):
    import ror.training as training
    from ror.experiment import run_experiment

    calls = {"n": 0}

    def budget_runs_out_after_start():
        calls["n"] += 1
        return 1e9 if calls["n"] == 1 else 0.0   # fine at start, exhausted after step 1

    monkeypatch.setattr(training, "seconds_left", budget_runs_out_after_start)
    runs = tmp_path / "runs"
    cfg = _cfg()
    assert run_experiment(cfg, runs_dir=runs) is None
    st = Registry(runs).load(cfg.experiment_id)
    assert st.status == Status.PENDING.value and st.attempts == 0
    assert "paused" in st.last_error
    assert list((runs / cfg.experiment_id / "checkpoints").glob("checkpoint-*"))

    monkeypatch.setattr(training, "seconds_left", lambda: None)  # next session
    res = run_experiment(cfg, runs_dir=runs)
    assert res is not None
    assert res.extra["train"]["resumed_from_step"] >= 1
    assert res.extra["train"]["global_steps"] == 4
