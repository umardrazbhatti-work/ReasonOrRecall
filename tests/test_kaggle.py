"""Tests for the Kaggle runtime helpers (no Kaggle needed)."""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.kaggle import ensure_data_dir, find_data_dir, restore_runs
from ror.registry import Registry, Status


def _dataset(root: Path) -> Path:
    d = root / "ror-data"
    (d / "processed" / "finqa").mkdir(parents=True)
    (d / "ror_data_manifest.json").write_text("{}", encoding="utf-8")
    return d


def test_find_data_dir_searches_nested_inputs(tmp_path):
    d = _dataset(tmp_path / "input" / "datasets" / "someone")
    assert find_data_dir([tmp_path / "input"]) == d
    assert find_data_dir([tmp_path / "input"], max_depth=1) is None


def test_ensure_data_dir_extracts_an_unpacked_zip(tmp_path):
    inp = tmp_path / "input" / "ror-data-upload"
    inp.mkdir(parents=True)
    with zipfile.ZipFile(inp / "ror-data.zip", "w") as z:
        z.writestr("ror_data_manifest.json", "{}")
        z.writestr("processed/finqa/test.jsonl", "")
    got = ensure_data_dir([tmp_path / "input"], extract_to=tmp_path / "extracted")
    assert got == tmp_path / "extracted"
    assert (got / "processed" / "finqa" / "test.jsonl").exists()


def _previous_output(root: Path) -> Path:
    runs = root / "runs"
    reg = Registry(runs)
    reg.mark_running("done1", "A5-done"); reg.mark_completed("done1")
    reg.mark_running("dead1", "A5-dead")                    # session killed mid-run
    (runs / "dead1" / "checkpoints" / "checkpoint-50").mkdir(parents=True)
    (runs / "dead1" / "checkpoints" / "checkpoint-50" / "adapter.bin").write_text("w")
    (runs / "results.jsonl").write_text('{"exp_id": "done1"}\n', encoding="utf-8")
    return runs


def test_restore_runs_merges_registry_and_marks_killed_runs(tmp_path):
    _previous_output(tmp_path / "input" / "my-notebook-v3")
    dest = tmp_path / "working" / "runs"
    rep = restore_runs(dest, roots=[tmp_path / "input"])
    assert rep["interrupted"] == ["dead1"] and rep["results_added"] == 1
    reg = Registry(dest, max_attempts=2)
    assert reg.should_run("done1") == (False, "already completed")
    st = reg.load("dead1")
    assert st.status == Status.FAILED.value and st.attempts == 1
    assert reg.should_run("dead1")[0]                         # one retry left
    assert (dest / "dead1" / "checkpoints" / "checkpoint-50" / "adapter.bin").exists()


def test_restore_runs_is_idempotent_and_never_overwrites(tmp_path):
    _previous_output(tmp_path / "input" / "nb")
    dest = tmp_path / "working" / "runs"
    restore_runs(dest, roots=[tmp_path / "input"])
    (dest / "done1" / "status.json").write_text(json.dumps({"exp_id": "done1", "name": "x",
                                                            "status": "completed"}))
    rep = restore_runs(dest, roots=[tmp_path / "input"])
    assert rep["files_copied"] == 0 and rep["results_added"] == 0
    assert json.loads((dest / "done1" / "status.json").read_text())["name"] == "x"
    assert len((dest / "results.jsonl").read_text().splitlines()) == 1


def test_runs_dir_for_uses_the_suites_runs_dir(tmp_path):
    from ror.kaggle import runs_dir_for

    root = Path(__file__).resolve().parents[1] / "configs"
    assert runs_dir_for(root / "suite_smoke.yaml", working=tmp_path) == tmp_path / "runs_smoke"
    assert runs_dir_for(root / "suite_phase1.yaml", working=tmp_path) == tmp_path / "runs"


def test_restore_never_mixes_smoke_and_study_registries(tmp_path):
    _previous_output(tmp_path / "input" / "nb")            # a "runs" registry
    rep = restore_runs(tmp_path / "working" / "runs_smoke", roots=[tmp_path / "input"])
    assert rep["sources"] == [] and rep["files_copied"] == 0
