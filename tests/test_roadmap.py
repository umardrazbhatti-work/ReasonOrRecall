"""Tests for the roadmap tracker (ror.roadmap)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.roadmap import load, parse, summary  # noqa: E402

SAMPLE = """# Roadmap
**Status: APPROVED v9 (2026-01-01).** More text.

| Phase | Goal | GPU (est.) | Status |
|---|---|---|---|
| **P0** Base | goal | ~0.5 h used | Done |
| **P1** Freeze | goal | ~3 h | Open |
| **P2** Data | goal | 0 | Open |
| **P3** Train | goal | ~64 h | Open |
| **P8** Paid | goal | ~15–20 h rented A100 | Open |

### P0 — Base  *(done)*
- [x] **P0.1** Framework: ids and registry.
### P1 — Freeze
- [x] **P1.1 (M) Train once, evaluate on every split.** Detail here.
- [~] **P1.2 (S)** Matching rule; more words.
- [ ] **P1.3 (C)** Stats
### P2 — Data  *(no GPU; parallel)*
- [ ] **P2.1 (M)** EDGAR fetcher: FY2025 10-Ks.
### P3 — Train  *(~64 GPU-h)*
- [ ] **P3.1 (M)** A5 seed 0
### P8 — Paid
- [ ] **P8.1 (M)** Preemption check

## 8. Run log
| Date | Results folder | Tasks | Outcome | GPU-h |
|---|---|---|---|---|
| 2026-01-02 | a | P0.5 | ok | 0.25 |
| 2026-01-03 | b | P1.1 | ok | 1.5 |

## 9. Decision log
| D1 | 2026-01-01 | x | yes |
"""


def test_parse_phases_tasks_and_ledger():
    rm = parse(SAMPLE)
    assert rm.status_line == "APPROVED v9 (2026-01-01)."
    assert [p.id for p in rm.phases] == ["P0", "P1", "P2", "P3", "P8"]
    p1 = rm.phase("P1")
    assert [(t.id, t.status, t.priority) for t in p1.tasks] == [
        ("P1.1", "done", "M"), ("P1.2", "doing", "S"), ("P1.3", "open", "C")]
    assert p1.tasks[0].title == "Train once, evaluate on every split"
    assert p1.tasks[1].title == "Matching rule"
    assert rm.phase("P2").tasks[0].title == "EDGAR fetcher: FY2025 10-Ks"
    assert rm.gpu_used_h == 1.75
    assert rm.gpu_budget_h == 67.0          # P1 + P2 + P3; not "used", not "rented"


def test_current_and_parallel_phases():
    rm = parse(SAMPLE)
    assert rm.current.id == "P1"            # P0 complete
    assert [p.id for p in rm.parallel()] == ["P2"]
    text = summary(rm)
    assert "current phase P1 Freeze" in text and "parallel phase P2 Data" in text
    assert "P1.2 (S) Matching rule [in progress]" in text
    assert "1.75 h used of ~67 h" in text


def test_the_real_roadmap_is_well_formed():
    rm = load()
    ids = [t.id for p in rm.phases for t in p.tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    assert [p.id for p in rm.phases] == [f"P{i}" for i in range(11)]
    assert all(p.tasks for p in rm.phases)
    assert all(t.id.startswith(p.id + ".") for p in rm.phases for t in p.tasks)
    assert rm.status_line.startswith("APPROVED")
    assert 200 <= rm.gpu_budget_h <= 300
