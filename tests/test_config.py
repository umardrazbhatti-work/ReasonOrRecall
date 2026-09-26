"""Tests for config loading and the deterministic experiment id."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.config import ExperimentConfig, config_from_dict, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_strings_are_coerced():
    # PyYAML reads `2e-4` (no dot) as a string; it must not leak into the config
    cfg = config_from_dict({"arm": "A5", "model": "qwen2.5-3b",
                            "learning_rate": "2e-4", "epochs": "3"})
    assert cfg.learning_rate == 2e-4 and isinstance(cfg.learning_rate, float)
    assert cfg.epochs == 3 and isinstance(cfg.epochs, int)
    same = config_from_dict({"arm": "A5", "model": "qwen2.5-3b",
                             "learning_rate": 2e-4, "epochs": 3})
    assert cfg.experiment_id == same.experiment_id


def test_unknown_keys_are_ignored():
    cfg = config_from_dict({"arm": "A5", "model": "qwen2.5-3b", "learning_rte": 1.0})
    assert cfg.learning_rate == ExperimentConfig(arm="A5", model="x").learning_rate


def test_id_is_deterministic_and_ignores_operational_fields():
    a = ExperimentConfig(arm="A5", model="qwen2.5-3b", notes="x", max_attempts=5)
    b = ExperimentConfig(arm="A5", model="qwen2.5-3b")
    assert a.experiment_id == b.experiment_id


def test_result_affecting_fields_change_the_id():
    base = ExperimentConfig(arm="A5", model="qwen2.5-3b")
    for field, value in [("temperature", 0.7), ("max_new_tokens", 64),
                         ("grad_accum", 8), ("batch_size", 2), ("lora_dropout", 0.1)]:
        other = ExperimentConfig(arm="A5", model="qwen2.5-3b", **{field: value})
        assert other.experiment_id != base.experiment_id, field


def test_experiment_yaml_merges_over_base():
    cfg = load_experiment_config(ROOT / "configs" / "experiments" / "A5_qlora_answer.yaml",
                                 base_yaml=ROOT / "configs" / "base.yaml")
    assert cfg.arm == "A5" and cfg.supervision == "answer"
    assert cfg.lora_r == 16  # from base.yaml


def test_training_id_ignores_split_and_decoding_only():
    from ror.config import ExperimentConfig

    base = ExperimentConfig(arm="A5", model="qwen2.5-3b", supervision="answer")
    same = [ExperimentConfig(arm="A5", model="qwen2.5-3b", supervision="answer", **kw)
            for kw in ({"split": "clean"}, {"max_new_tokens": 64}, {"eval_examples": 10},
                       {"infer_batch_size": 2})]
    assert all(c.training_id == base.training_id for c in same)
    assert all(c.experiment_id != base.experiment_id for c in same[:3])
    for kw in ({"seed": 1}, {"learning_rate": 1e-4}, {"epochs": 2}, {"model": "qwen2.5-7b"}):
        c = ExperimentConfig(**{"arm": "A5", "model": "qwen2.5-3b", "supervision": "answer", **kw})
        assert c.training_id != base.training_id, kw
