"""Experiment registry — the piece that makes runs idempotent and stops failed
experiments from being retried forever.

Design: one small JSON status file per experiment at
`runs/<exp_id>/status.json`. No global lock, no database — a scan of `runs/`
reconstructs the full picture, which is robust across killed Kaggle sessions and
easy to inspect or hand-edit.

Status lifecycle:

    (new) --run--> RUNNING --ok--> COMPLETED           (skipped forever after)
                      |--error--> FAILED --retry--> RUNNING ...
                      |            (after max_attempts) --> PERMANENTLY_FAILED
                      |                                     (skipped forever)
                      '--(process died, stale)--> retryable

`should_run` is the one method callers need: it returns (run?, reason).
"""
from __future__ import annotations

import enum
import json
import os
import socket
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


class Status(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"


TERMINAL_SKIP = {Status.COMPLETED.value, Status.PERMANENTLY_FAILED.value}


@dataclass
class RunState:
    exp_id: str
    name: str
    status: str = Status.PENDING.value
    attempts: int = 0
    max_attempts: int = 2
    first_seen: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_error: Optional[str] = None
    host: Optional[str] = None
    pid: Optional[int] = None
    metrics_path: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "RunState":
        d = json.loads(text)
        known = {f for f in RunState.__dataclass_fields__}  # type: ignore[attr-defined]
        return RunState(**{k: v for k, v in d.items() if k in known})


class Registry:
    """Tracks experiment status under a runs directory."""

    def __init__(
        self,
        runs_dir: str | Path = "runs",
        max_attempts: int = 2,
        stale_running_secs: int = 6 * 3600,
    ):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.stale_running_secs = stale_running_secs

    # --- paths ---
    def run_dir(self, exp_id: str) -> Path:
        d = self.runs_dir / exp_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_path(self, exp_id: str) -> Path:
        return self.runs_dir / exp_id / "status.json"

    # --- io ---
    def load(self, exp_id: str) -> Optional[RunState]:
        p = self._state_path(exp_id)
        if not p.exists():
            return None
        try:
            return RunState.from_json(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, state: RunState) -> None:
        state.last_update = time.time()
        p = self._state_path(state.exp_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(state.to_json(), encoding="utf-8")
        os.replace(tmp, p)  # atomic

    # --- the decision ---
    def should_run(self, exp_id: str, name: str = "") -> tuple[bool, str]:
        st = self.load(exp_id)
        if st is None:
            return True, "new"
        if st.status == Status.COMPLETED.value:
            return False, "already completed"
        if st.status == Status.PERMANENTLY_FAILED.value:
            return False, f"permanently failed after {st.attempts} attempt(s)"
        if st.status == Status.RUNNING.value:
            age = time.time() - st.last_update
            if age > self.stale_running_secs:
                if st.attempts < st.max_attempts:
                    return True, f"stale running ({int(age)}s) — retrying"
                return False, "stale running but out of attempts"
            return False, f"currently running on {st.host or '?'} (pid {st.pid})"
        if st.status == Status.FAILED.value:
            if st.attempts < st.max_attempts:
                return True, f"retry {st.attempts + 1}/{st.max_attempts}"
            return False, "out of retry attempts"
        if st.status == Status.PENDING.value:
            return True, "pending"
        return True, "unknown status — running"

    # --- transitions ---
    def mark_running(self, exp_id: str, name: str = "") -> RunState:
        st = self.load(exp_id) or RunState(exp_id=exp_id, name=name, max_attempts=self.max_attempts)
        st.name = name or st.name
        st.status = Status.RUNNING.value
        st.attempts += 1
        st.host = socket.gethostname()
        st.pid = os.getpid()
        self.save(st)
        return st

    def mark_completed(self, exp_id: str, metrics_path: str | None = None) -> RunState:
        st = self.load(exp_id) or RunState(exp_id=exp_id, name="", max_attempts=self.max_attempts)
        st.status = Status.COMPLETED.value
        st.last_error = None
        st.metrics_path = metrics_path
        self.save(st)
        return st

    def mark_failed(self, exp_id: str, error: str) -> RunState:
        st = self.load(exp_id) or RunState(exp_id=exp_id, name="", max_attempts=self.max_attempts)
        st.last_error = (error or "")[:4000]
        if st.attempts >= st.max_attempts:
            st.status = Status.PERMANENTLY_FAILED.value
        else:
            st.status = Status.FAILED.value
        self.save(st)
        return st

    def mark_not_ready(self, exp_id: str, reason: str) -> RunState:
        """Undo the attempt counted by `mark_running` and return to PENDING.

        For runs that could not start because the *code* is not ready (a stub
        still raising NotImplementedError), not because the experiment failed.
        Without this, launching a suite before the ML modules exist would burn
        every experiment's attempts and leave them permanently failed.
        """
        st = self.load(exp_id) or RunState(exp_id=exp_id, name="", max_attempts=self.max_attempts)
        st.attempts = max(0, st.attempts - 1)
        st.status = Status.PENDING.value
        st.last_error = f"not ready: {reason}"[:4000]
        self.save(st)
        return st

    def force_reset(self, exp_id: str) -> None:
        """Clear state so the experiment runs fresh. Only on explicit request."""
        st = self.load(exp_id)
        if st:
            st.status = Status.PENDING.value
            st.attempts = 0
            st.last_error = None
            self.save(st)

    # --- views ---
    def all_states(self) -> list[RunState]:
        out = []
        for d in sorted(self.runs_dir.glob("*/")):
            st = self.load(d.name)
            if st:
                out.append(st)
        return out

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for st in self.all_states():
            counts[st.status] = counts.get(st.status, 0) + 1
        return counts
