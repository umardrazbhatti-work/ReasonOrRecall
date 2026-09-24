"""Tests for the suite planner and the results aggregator (scripts/)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dry_run(tmp_path, *extra: str) -> str:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_suite.py"),
         str(ROOT / "configs" / "suite_phase1.yaml"), "--dry-run",
         "--runs-dir", str(tmp_path), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return out.stdout + out.stderr


def test_suite_filters_narrow_to_one_experiment(tmp_path):
    log = _dry_run(tmp_path, "--only", "A5", "--model", "qwen2.5-3b",
                   "--split", "standard", "--seed", "0")
    assert "1 experiments" in log
    assert "A5-qwen2.5-3b-finqa-standard-s0" in log


def test_phase1_suite_never_plans_70b():
    run_suite = _load_script("run_suite")
    configs, _ = run_suite.expand(str(ROOT / "configs" / "suite_phase1.yaml"))
    assert configs and not any("72b" in c.model for c in configs)


def _row(exp_id, arm="A5", split="standard", seed=0, em=0.5, ts=0.0, dataset="finqa", **cfg):
    config = {"arm": arm, "model": "m", "split": split, "seed": seed, "epochs": 3, **cfg}
    return {"exp_id": exp_id, "arm": arm, "model": "m", "split": split, "seed": seed,
            "dataset": dataset, "exact_match": em, "timestamp": ts, "config": config}


def test_aggregate_dedupes_reruns_and_pairs_splits():
    agg = _load_script("aggregate_results")
    rows = [
        _row("a", em=0.10, ts=1), _row("a", em=0.60, ts=2),   # forced re-run: latest wins
        _row("b", seed=1, em=0.80, ts=1),                    # a real second seed
        _row("c", split="clean", em=0.40, ts=1),
    ]
    table, _ = agg.aggregate(rows)
    assert len(table) == 1
    row = table[0]
    assert row["n_seeds_standard"] == 2
    assert abs(row["exact_match_standard"] - 0.70) < 1e-9     # mean(0.60, 0.80)
    assert abs(row["contamination_gap"] - 0.30) < 1e-9


def test_aggregate_separates_datasets_and_variants():
    agg = _load_script("aggregate_results")
    rows = [_row("a", dataset="finqa"), _row("b", dataset="tatqa"),
            _row("c", epochs=1)]
    table, _ = agg.aggregate(rows)
    assert len(table) == 3
