"""Tests for the registry — the never-repeat-finished-experiments logic."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.registry import Registry, Status


def test_new_experiment_runs(tmp_path):
    reg = Registry(tmp_path)
    run, reason = reg.should_run("abc", "exp")
    assert run and reason == "new"


def test_completed_is_skipped(tmp_path):
    reg = Registry(tmp_path)
    reg.mark_running("abc", "exp")
    reg.mark_completed("abc")
    run, reason = reg.should_run("abc", "exp")
    assert not run
    assert "completed" in reason


def test_failure_retries_then_stops(tmp_path):
    reg = Registry(tmp_path, max_attempts=2)
    # attempt 1
    reg.mark_running("abc", "exp")
    reg.mark_failed("abc", "boom")
    run, _ = reg.should_run("abc", "exp")
    assert run  # one retry allowed
    # attempt 2 -> hits max_attempts -> permanently failed
    reg.mark_running("abc", "exp")
    reg.mark_failed("abc", "boom again")
    st = reg.load("abc")
    assert st.status == Status.PERMANENTLY_FAILED.value
    run, reason = reg.should_run("abc", "exp")
    assert not run
    assert "permanently failed" in reason


def test_running_is_skipped_but_stale_retries(tmp_path):
    reg = Registry(tmp_path, max_attempts=3, stale_running_secs=1)
    reg.mark_running("abc", "exp")
    run, _ = reg.should_run("abc", "exp")
    assert not run  # fresh running -> skip
    time.sleep(1.1)
    run, reason = reg.should_run("abc", "exp")
    assert run and "stale" in reason


def test_force_reset(tmp_path):
    reg = Registry(tmp_path)
    reg.mark_running("abc", "exp")
    reg.mark_completed("abc")
    reg.force_reset("abc")
    run, _ = reg.should_run("abc", "exp")
    assert run


def test_not_ready_does_not_consume_an_attempt(tmp_path):
    reg = Registry(tmp_path, max_attempts=2)
    for _ in range(3):  # far more launches than max_attempts
        reg.mark_running("abc", "exp")
        reg.mark_not_ready("abc", "NotImplementedError: stub")
    st = reg.load("abc")
    assert st.status == Status.PENDING.value and st.attempts == 0
    run, _ = reg.should_run("abc", "exp")
    assert run


def test_summary_counts(tmp_path):
    reg = Registry(tmp_path)
    reg.mark_running("a", "a"); reg.mark_completed("a")
    reg.mark_running("b", "b")
    s = reg.summary()
    assert s.get("completed") == 1
    assert s.get("running") == 1
