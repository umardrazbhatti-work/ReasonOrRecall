"""Tests for the experiment runner's failure accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ror.experiment as experiment
from ror.config import ExperimentConfig
from ror.registry import Registry, Status

ROR_DIR = Path(experiment.__file__).resolve().parent


def _fn_raising_from(filename: Path):
    """A function whose NotImplementedError appears to come from `filename`."""
    ns: dict = {}
    code = "def f(cfg, run_dir):\n    raise NotImplementedError('stub')\n"
    exec(compile(code, str(filename), "exec"), ns)
    return ns["f"]


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(arm="A5", model="qwen2.5-3b", supervision="answer")


def test_stub_not_implemented_does_not_count_as_attempt(tmp_path, monkeypatch, fake_data_dir):
    monkeypatch.setattr(experiment, "_execute", _fn_raising_from(ROR_DIR / "fake_stub.py"))
    cfg = _cfg()
    reg = Registry(tmp_path, max_attempts=2)
    for _ in range(3):
        assert experiment.run_experiment(cfg, runs_dir=tmp_path, registry=reg) is None
    st = reg.load(cfg.experiment_id)
    assert st.status == Status.PENDING.value and st.attempts == 0
    assert (tmp_path / cfg.experiment_id / "error.txt").exists()


def test_library_not_implemented_counts_as_failure(tmp_path, monkeypatch, fake_data_dir):
    monkeypatch.setattr(experiment, "_execute", _fn_raising_from(tmp_path / "somelib.py"))
    cfg = _cfg()
    reg = Registry(tmp_path, max_attempts=2)
    experiment.run_experiment(cfg, runs_dir=tmp_path, registry=reg)
    experiment.run_experiment(cfg, runs_dir=tmp_path, registry=reg)
    st = reg.load(cfg.experiment_id)
    assert st.status == Status.PERMANENTLY_FAILED.value and st.attempts == 2


def test_completed_run_records_full_config(tmp_path, monkeypatch, fake_data_dir):
    from ror.results import RunResult, load_results

    def fake_execute(cfg, run_dir):
        return RunResult(exp_id=cfg.experiment_id, name=cfg.resolved_name(),
                         arm=cfg.arm, model=cfg.model, exact_match=0.5)

    monkeypatch.setattr(experiment, "_execute", fake_execute)
    cfg = _cfg()
    experiment.run_experiment(cfg, runs_dir=tmp_path)
    rows = load_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["config"]["learning_rate"] == cfg.learning_rate
    assert rows[0]["config"]["lora_r"] == cfg.lora_r
    # finished -> skipped next time
    assert experiment.run_experiment(cfg, runs_dir=tmp_path) is None
    assert len(load_results(tmp_path)) == 1


def test_missing_data_skips_without_touching_the_registry(tmp_path, fake_data_dir):
    cfg = ExperimentConfig(arm="A5", model="qwen2.5-3b", supervision="answer", split="clean")
    reg = Registry(tmp_path / "runs")
    assert experiment.run_experiment(cfg, runs_dir=tmp_path / "runs", registry=reg) is None
    assert reg.load(cfg.experiment_id) is None  # no attempt, no state


def test_session_budget_pauses_before_training_without_an_attempt(tmp_path, monkeypatch,
                                                                  fake_data_dir):
    import ror.training as training

    monkeypatch.setattr(training, "seconds_left", lambda: 60.0)   # 1 minute left
    cfg = _cfg()
    reg = Registry(tmp_path)
    assert experiment.run_experiment(cfg, runs_dir=tmp_path, registry=reg) is None
    st = reg.load(cfg.experiment_id)
    assert st.status == Status.PENDING.value and st.attempts == 0
    assert "paused" in st.last_error
