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
    assert res.exact_match_strict is not None and res.exact_match_strict <= res.exact_match
    assert res.train_flops > 0 and res.infer_flops > 0
    gpu = res.extra["gpu"]                                 # P1.6: GPU time for the cost
    assert gpu["n_gpus"] == 0 and gpu["train_seconds"] > 0 and gpu["eval_seconds"] > 0
    assert res.cost_usd is None                            # CPU: no priced GPU
    assert res.extra["train"]["n_train"] == 2 and res.extra["train"]["global_steps"] == 4
    assert res.extra["train"]["adapter_dtype"] == ["float32"]

    run_dir = runs / cfg.experiment_id
    home = runs / "_adapters" / cfg.training_id          # shared by every split
    assert (home / "adapter" / "adapter_config.json").exists()
    assert not (home / "checkpoints").exists()           # removed once the adapter exists
    assert json.loads((run_dir / "training.json").read_text())["reused"] is False
    sel = res.extra["train"]["selection"]                 # P1.3: best of 4 dev checks
    assert [c["step"] for c in sel["checks"]] == [1, 2, 3, 4]
    assert sel["chosen_step"] in (1, 2, 3, 4) and sel["dev_examples"] == 2
    assert not (home / "selection").exists() and not (home / "train_done.json").exists()
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
    assert list((runs / "_adapters" / cfg.training_id / "checkpoints").glob("checkpoint-*"))

    monkeypatch.setattr(training, "seconds_left", lambda: None)  # next session
    res = run_experiment(cfg, runs_dir=runs)
    assert res is not None
    assert res.extra["train"]["resumed_from_step"] >= 1
    assert res.extra["train"]["global_steps"] == 4


def test_lora_weights_stay_fp32_when_the_model_is_4bit(tmp_path, tiny_model):
    # trl casts a quantized model's LoRA weights to bf16 when the trainer is built;
    # the fp16 GradScaler on a T4 cannot unscale bf16 gradients. Flag the tiny
    # model as 4-bit to take that code path, then apply ror.training's fix.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    from ror.models import load_student
    from ror.training import _adapters_fp32

    lm = load_student("qwen2.5-3b")
    lm.model.is_loaded_in_4bit = True           # what a bitsandbytes 4-bit load sets
    model = get_peft_model(lm.model, LoraConfig(r=4, target_modules=["q_proj"],
                                                task_type="CAUSAL_LM"))
    rec = {"prompt": [{"role": "user", "content": "1+1?"}],
           "completion": [{"role": "assistant", "content": "2"}]}
    trainer = SFTTrainer(model=model, processing_class=lm.tokenizer,
                         train_dataset=Dataset.from_list([rec] * 2),
                         args=SFTConfig(output_dir=str(tmp_path), use_cpu=True,
                                        report_to="none", max_length=64))
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    assert _adapters_fp32(trainer.model) >= 0
    assert trainable and all(p.dtype == torch.float32 for p in trainable)
    assert _adapters_fp32(trainer.model) == 0   # idempotent


def test_trained_once_and_evaluated_on_another_split(tmp_path, tiny_model, monkeypatch):
    # roadmap P1.1: the clean/control evaluations must reuse the standard run's adapter
    import ror.training as training
    from ror.experiment import run_experiment

    runs = tmp_path / "runs"
    std = _cfg(epochs=1)
    other = _cfg(epochs=1, split="dev")
    assert std.training_id == other.training_id and std.experiment_id != other.experiment_id
    assert run_experiment(std, runs_dir=runs) is not None

    def must_not_train(*a, **k):
        raise AssertionError("retrained an adapter that already exists")

    monkeypatch.setattr(training, "build_sft_records", must_not_train)
    res = run_experiment(other, runs_dir=runs)
    assert res is not None and res.extra["train"]["reused_adapter"] is True
    assert res.train_flops == res.extra["train"]["train_flops"] > 0   # still costs its training
    assert len(list((runs / "_adapters").iterdir())) == 1
    assert Registry(runs).load(other.experiment_id).status == Status.COMPLETED.value


def test_selection_steps_are_even_and_end_at_the_last_step():
    from ror.training import selection_steps

    assert selection_steps(385, 4) == [97, 194, 291, 385]
    assert selection_steps(4, 4) == [1, 2, 3, 4]
    assert selection_steps(3, 4) == [1, 2, 3]
    assert selection_steps(10, 1) == [10]


def test_session_ending_between_training_and_selection(tmp_path, tiny_model, monkeypatch):
    # training finishes, too little time is left to select: pause (no attempt used);
    # the next session selects without retraining
    import time

    import ror.training as training
    from ror.experiment import run_experiment

    runs = tmp_path / "runs"
    cfg = _cfg(epochs=1)
    monkeypatch.setenv("ROR_DEADLINE_UNIX", str(time.time() + 3600))
    monkeypatch.setattr(training, "TRAIN_STOP_RESERVE_S", 0)
    monkeypatch.setattr(training, "MIN_SELECTION_S", 7200)
    assert run_experiment(cfg, runs_dir=runs) is None
    st = Registry(runs).load(cfg.experiment_id)
    assert st.status == Status.PENDING.value and st.attempts == 0 and "select" in st.last_error
    home = runs / "_adapters" / cfg.training_id
    assert (home / "train_done.json").exists() and list((home / "selection").iterdir())

    monkeypatch.delenv("ROR_DEADLINE_UNIX")

    def must_not_train(*a, **k):
        raise AssertionError("retrained after training had finished")

    monkeypatch.setattr(training, "build_sft_records", must_not_train)
    res = run_experiment(cfg, runs_dir=runs)
    assert res is not None and res.extra["train"]["selection"]["chosen_step"] in (1, 2)


def test_multi_item_batches_without_gradient_checkpointing(tmp_path, tiny_model):
    # the P1.4 pilot's settings: batch > 1 (grouped by length), checkpointing off
    from ror.experiment import run_experiment

    cfg = _cfg(epochs=1, batch_size=2, grad_accum=1, gradient_checkpointing=False)
    res = run_experiment(cfg, runs_dir=tmp_path / "runs")
    assert res is not None
    assert res.extra["train"]["batch"] == {"batch_size": 2, "grad_accum": 1,
                                           "gradient_checkpointing": False}
    assert res.extra["train"]["global_steps"] == 1
