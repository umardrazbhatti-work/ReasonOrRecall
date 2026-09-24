"""Tests for the sandboxed Program-of-Thoughts executor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.sandbox import run_program


def test_simple_program():
    r = run_program("rev18 = 1200\nrev19 = 1380\nanswer = (rev19 - rev18)/rev18*100")
    assert r.ok
    assert abs(float(r.value) - 15.0) < 1e-6


def test_missing_answer_variable():
    r = run_program("x = 1 + 1")  # no `answer`
    assert not r.ok


def test_runtime_error_is_caught():
    r = run_program("answer = 1/0")
    assert not r.ok
    assert not r.timed_out


def test_timeout():
    r = run_program("while True:\n    pass\nanswer = 1", timeout=1.0)
    assert not r.ok
    assert r.timed_out


def test_string_answer():
    r = run_program("answer = 'increase'")
    assert r.ok and r.value == "increase"
