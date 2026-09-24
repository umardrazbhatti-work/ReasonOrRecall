"""Sandboxed executor for model-generated Program-of-Thoughts (PoT) code.

A PoT program is Python that must assign a variable named ``answer``. We run it
in a **separate subprocess** with a wall-clock timeout and (on POSIX) CPU/memory
limits, capture the value of ``answer``, and return it. Model output is untrusted,
so isolation matters.

This uses subprocess + `python -I` (isolated mode) + rlimits. For stronger
isolation with fully untrusted teachers, wrap the subprocess in nsjail/firejail
or a container — do not weaken what is here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Harness appended to the program: capture `answer` and print it as JSON.
_HARNESS = """
import json as _json, sys as _sys
try:
    _ans = answer  # noqa: F821  (the program must define `answer`)
except NameError:
    _sys.stderr.write("NO_ANSWER_VARIABLE")
    _sys.exit(3)
try:
    print("__RESULT__" + _json.dumps({"answer": _ans}))
except TypeError:
    print("__RESULT__" + _json.dumps({"answer": repr(_ans)}))
"""

# Best-effort resource limits inside the child (POSIX only).
_PREEXEC = """
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    _mem = {mem_bytes}
    resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
except Exception:
    pass
"""


@dataclass
class ExecResult:
    ok: bool
    value: Any = None
    error: str = ""
    timed_out: bool = False


def run_program(
    code: str,
    timeout: float = 5.0,
    cpu_secs: int = 5,
    mem_mb: int = 512,
) -> ExecResult:
    """Execute a PoT program and return the captured ``answer``.

    Returns ExecResult(ok=False, ...) on any error, timeout, or missing answer.
    Never raises for ordinary program failures.
    """
    if not code or "answer" not in code:
        return ExecResult(ok=False, error="program does not assign `answer`")

    preamble = _PREEXEC.format(cpu=cpu_secs, mem_bytes=mem_mb * 1024 * 1024)
    full = preamble + "\n" + code + "\n" + _HARNESS

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "prog.py"
        script.write_text(full)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env={"PYTHONHASHSEED": "0"},  # minimal env; no inherited secrets
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error="timeout", timed_out=True)

        if proc.returncode != 0:
            err = (proc.stderr or "").strip()[:1000] or f"exit {proc.returncode}"
            return ExecResult(ok=False, error=err)

        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                payload = json.loads(line[len("__RESULT__"):])
                return ExecResult(ok=True, value=payload["answer"])
        return ExecResult(ok=False, error="no result marker in output")
