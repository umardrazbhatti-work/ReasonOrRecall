"""Progress tracking against the approved roadmap (docs/ROADMAP.md).

The roadmap is plain Markdown so both people and code can read it. This module
parses it: phases (`### P3 — ...`), tasks (`- [x] **P3.1 (M)** ...`), the GPU
estimates in the phases-at-a-glance table and the run log's GPU-hours, and
reports progress, the current phase and the next tasks.

Task status marks: `[x]` done, `[~]` in progress, `[ ]` open.
Priority marks: (M) must, (S) should, (C) could.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .paths import repo_root

ROADMAP_FILE = "docs/ROADMAP.md"

# Phases that run alongside others (roadmap §6): phase -> phases it may overlap.
PARALLEL = {"P2": ("P1", "P3", "P4", "P5")}

_PHASE = re.compile(r"^### (P\d+) — (.+?)(?:\s{2,}\*\(.*\)\*)?\s*$")
_TASK = re.compile(r"^- \[([ x~])\] \*\*(P\d+\.\d+)(?: \(([MSC])\))?(.*?)\*\*(.*)$")
_GLANCE = re.compile(r"^\| \*\*(P\d+)\*\* [^|]*\|[^|]*\|\s*([^|]*?)\s*\|")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Task:
    id: str
    status: str                  # "done" | "doing" | "open"
    priority: Optional[str]      # "M" | "S" | "C" | None
    title: str


@dataclass
class Phase:
    id: str
    name: str
    tasks: list[Task] = field(default_factory=list)
    gpu_estimate: str = ""

    @property
    def done(self) -> int:
        return sum(t.status == "done" for t in self.tasks)

    @property
    def complete(self) -> bool:
        return bool(self.tasks) and self.done == len(self.tasks)


@dataclass
class Roadmap:
    status_line: str
    phases: list[Phase]
    gpu_used_h: float
    gpu_budget_h: float

    def phase(self, pid: str) -> Phase:
        return next(p for p in self.phases if p.id == pid)

    @property
    def current(self) -> Optional[Phase]:
        """The first phase with open tasks (its gate has not passed)."""
        return next((p for p in self.phases if not p.complete), None)

    def parallel(self) -> list[Phase]:
        """Open phases that may run alongside the current one."""
        cur = self.current
        if cur is None:
            return []
        return [p for p in self.phases if p is not cur and not p.complete
                and cur.id in PARALLEL.get(p.id, ())]


def _title(inside_bold: str, after_bold: str) -> str:
    text = (inside_bold.strip() or after_bold.strip()).replace("**", "").strip()
    first = re.split(r"(?<=[.;])\s", text, maxsplit=1)[0].rstrip(".;")
    return first if len(first) <= 90 else first[:87] + "..."


def parse(text: str) -> Roadmap:
    """Parse the roadmap Markdown."""
    m = re.search(r"\*\*Status:\s*(.*?)\*\*", text)
    status_line = m.group(1).strip() if m else ""
    phases: list[Phase] = []
    estimates: dict[str, str] = {}
    run_log_hours = 0.0
    in_run_log = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_run_log = line.lower().startswith("## 8. run log")
        m = _PHASE.match(line)
        if m:
            phases.append(Phase(m.group(1), m.group(2).strip()))
            continue
        m = _TASK.match(line)
        if m and phases:
            status = {"x": "done", "~": "doing", " ": "open"}[m.group(1)]
            phases[-1].tasks.append(Task(m.group(2), status, m.group(3),
                                         _title(m.group(4), m.group(5))))
            continue
        m = _GLANCE.match(line)
        if m:
            estimates[m.group(1)] = m.group(2)
            continue
        if in_run_log and line.startswith("| 20"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            try:
                run_log_hours += float(cells[-1])
            except ValueError:
                pass
    budget = 0.0
    for p in phases:
        p.gpu_estimate = estimates.get(p.id, "")
        est = p.gpu_estimate.lower()
        n = _NUMBER.search(est.replace("–", "-"))
        if n and "used" not in est and "rented" not in est:   # Kaggle budget only
            budget += float(n.group())
    return Roadmap(status_line, phases, run_log_hours, budget)


def load(path: Optional[str | Path] = None) -> Roadmap:
    """Parse docs/ROADMAP.md (or `path`)."""
    p = Path(path) if path else repo_root() / ROADMAP_FILE
    return parse(p.read_text(encoding="utf-8"))


def summary(rm: Roadmap, next_n: int = 3) -> str:
    """Human-readable progress report."""
    lines = [f"ROADMAP  {rm.status_line}", ""]
    width = max(len(f"{p.id} {p.name}") for p in rm.phases)
    for p in rm.phases:
        n = len(p.tasks)
        bar = "#" * round(10 * p.done / n) if n else ""
        mark = "done" if p.complete else ("..." if p is rm.current else "")
        lines.append(f"  {f'{p.id} {p.name}':<{width}}  {p.done:>2}/{n:<2} "
                     f"[{bar:<10}] {mark}")
    lines.append("")
    cur = rm.current
    if cur is None:
        lines.append("All phases complete.")
    else:
        for ph, label in [(cur, "current")] + [(p, "parallel") for p in rm.parallel()]:
            nxt = [t for t in ph.tasks if t.status != "done"][:next_n]
            lines.append(f"{label} phase {ph.id} {ph.name} - next:")
            for t in nxt:
                pr = f"({t.priority}) " if t.priority else ""
                state = " [in progress]" if t.status == "doing" else ""
                lines.append(f"  {t.id} {pr}{t.title}{state}")
    pct = 100 * rm.gpu_used_h / rm.gpu_budget_h if rm.gpu_budget_h else 0
    lines += ["", f"Kaggle GPU: {rm.gpu_used_h:.2f} h used of ~{rm.gpu_budget_h:.0f} h "
                  f"budget ({pct:.1f}%)"]
    return "\n".join(lines)
