"""Sandboxed executor for model-generated Program-of-Thoughts (PoT) code.

A PoT program is Python that must assign a variable named ``answer``. We run it
in a **separate subprocess** with a wall-clock timeout and (on POSIX) CPU/memory
limits, capture the value of ``answer``, and return it. Model output is untrusted,
so isolation matters.

Layers: subprocess + `python -I -B` (isolated mode, no bytecode writes) +
rlimits (POSIX) + an audit hook installed before the program runs that blocks
file writes, reads outside the Python installation, directory listings outside
it, network sockets, subprocesses/exec, and filesystem mutation (proposal 5.4:
"no file or network access"). The hook stops accidental or naive side effects
of model code; it is not a boundary against deliberately adversarial code. For
that, wrap the subprocess in nsjail/firejail or a container — do not weaken
what is here.
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

# Audit-hook guard (not .format()-ed). Everything the hook needs is bound as a
# default argument so the program cannot disable it by rebinding globals.
_GUARD = r"""
import os as _os, sys as _sys
def _install_guard():
    def norm(p, os_=_os):
        return os_.path.normcase(os_.path.abspath(os_.fsdecode(p)))
    roots = {_sys.prefix, _sys.base_prefix, _sys.exec_prefix, _sys.base_exec_prefix}
    roots.update(p for p in _sys.path if p)
    allowed = tuple(sorted({norm(p) for p in roots}))
    write_flags = _os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_APPEND | _os.O_TRUNC
    blocked = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn",
               "os.posix_spawn", "os.fork", "os.forkpty", "os.kill", "os.startfile",
               "os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir",
               "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.symlink",
               "os.link", "os.putenv", "os.unsetenv", "shutil.", "ctypes.",
               "webbrowser.", "winreg.", "_winapi.", "msvcrt.")

    def guard(event, args, norm=norm, allowed=allowed, write_flags=write_flags,
              blocked=blocked):
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return  # an already-open descriptor (e.g. stdout)
            writing = (any(c in mode for c in "wax+") if mode is not None
                       else bool(flags & write_flags))
            if writing or not norm(path).startswith(allowed):
                raise PermissionError("sandbox: file access blocked")
        elif event in ("os.listdir", "os.scandir"):
            path = args[0] if args and args[0] is not None else "."
            if not isinstance(path, int) and not norm(path).startswith(allowed):
                raise PermissionError("sandbox: directory listing blocked")
        elif event.startswith(blocked):
            raise PermissionError("sandbox: blocked " + event)

    _sys.addaudithook(guard)
_install_guard()
del _install_guard, _os, _sys
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
    full = preamble + "\n" + _GUARD + "\n" + code + "\n" + _HARNESS

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "prog.py"
        script.write_text(full, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env={"PYTHONHASHSEED": "0"},  # minimal env; no inherited secrets
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error="timeout", timed_out=True)

        if proc.returncode != 0:
            # keep the tail: the exception message is the last traceback line
            err = (proc.stderr or "").strip()[-1000:] or f"exit {proc.returncode}"
            return ExecResult(ok=False, error=err)

        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                payload = json.loads(line[len("__RESULT__"):])
                return ExecResult(ok=True, value=payload["answer"])
        return ExecResult(ok=False, error="no result marker in output")
