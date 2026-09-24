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


def test_stdlib_imports_still_work():
    r = run_program("import heapq, statistics\n"
                    "answer = heapq.nsmallest(1, [3, 1])[0] + statistics.mean([1, 3])")
    assert r.ok and r.value == 3


def test_file_write_is_blocked(tmp_path):
    target = tmp_path / "out.txt"
    r = run_program(f"open({str(target)!r}, 'w').write('x')\nanswer = 1")
    assert not r.ok and "sandbox" in r.error
    assert not target.exists()


def test_file_read_outside_python_is_blocked(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token")
    r = run_program(f"answer = open({str(secret)!r}).read()")
    assert not r.ok and "sandbox" in r.error


def test_network_and_processes_are_blocked():
    for code in ("import socket\nsocket.socket()\nanswer = 1",
                 "import subprocess\nsubprocess.run(['python', '-c', '0'])\nanswer = 1",
                 "import os\nos.system('echo hi')\nanswer = 1"):
        r = run_program(code)
        assert not r.ok, code
        assert "sandbox" in r.error or "PermissionError" in r.error, code


def test_filesystem_mutation_is_blocked(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    r = run_program(f"import os\nos.remove({str(victim)!r})\nanswer = 1")
    assert not r.ok and victim.exists()
