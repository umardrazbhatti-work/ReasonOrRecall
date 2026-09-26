"""Tests for the suite planner and the results aggregator (scripts/)."""
import importlib.util
import os
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


def _dry_run(tmp_path, *extra: str, data_dir=None) -> str:
    env = dict(os.environ)
    if data_dir is not None:
        env["ROR_DATA_DIR"] = str(data_dir)
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_suite.py"),
         str(ROOT / "configs" / "suite_phase1.yaml"), "--dry-run",
         "--runs-dir", str(tmp_path), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        env=env)
    return out.stdout + out.stderr


def test_suite_filters_narrow_to_one_experiment(tmp_path, fake_data_dir):
    log = _dry_run(tmp_path / "runs", "--only", "A5", "--model", "qwen2.5-3b",
                   "--split", "standard", "--seed", "0", data_dir=fake_data_dir)
    assert "1 experiments (1 to run, 0 skipped)" in log
    assert "A5-qwen2.5-3b-finqa-standard-s0" in log


def test_suite_plans_missing_clean_set_as_skipped(tmp_path, fake_data_dir):
    log = _dry_run(tmp_path / "runs", "--only", "A5", "--model", "qwen2.5-3b",
                   "--split", "clean", "--seed", "0", data_dir=fake_data_dir)
    assert "(0 to run, 1 skipped)" in log and "clean set not built" in log


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


def test_smoke_suite_applies_overrides_and_its_own_runs_dir():
    run_suite = _load_script("run_suite")
    configs, suite = run_suite.expand(str(ROOT / "configs" / "suite_smoke.yaml"))
    assert len(configs) == 1 and suite["runs_dir"] == "runs_smoke"
    c = configs[0]
    assert (c.arm, c.train_examples, c.eval_examples, c.epochs) == ("A5", 64, 32, 1)
    assert c.max_seq_len == 2048                     # base.yaml still applies


def test_pilot_suite_expands_its_variants():
    run_suite = _load_script("run_suite")
    configs, suite = run_suite.expand(str(ROOT / "configs" / "suite_pilot.yaml"))
    assert suite["runs_dir"] == "runs_pilot" and len(configs) == 4
    settings = [(c.batch_size, c.grad_accum, c.gradient_checkpointing) for c in configs]
    assert settings == [(1, 16, True), (4, 4, True), (1, 16, False), (4, 4, False)]
    assert all(c.batch_size * c.grad_accum == 16 for c in configs)      # same effective batch
    assert len({c.experiment_id for c in configs}) == 4
    assert configs[0].resolved_name().endswith("-b1x16-gc")
    assert all(c.train_examples == 256 and c.selection_checks == 1 for c in configs)


def test_aggregate_adds_confidence_intervals(tmp_path):
    agg = _load_script("aggregate_results")
    rows = [_row("a", em=0.5, ts=1), _row("b", seed=1, em=0.5, ts=1),
            _row("c", split="clean", em=0.25, ts=1)]
    items = {"a": {f"q{i}": i % 2 == 0 for i in range(200)},
             "b": {f"q{i}": i % 2 == 0 for i in range(200)},
             "c": {f"k{i}": i % 4 == 0 for i in range(200)}}
    table, _ = agg.aggregate(rows, items)
    row = table[0]
    assert row["exact_match_lo_standard"] < 0.5 < row["exact_match_hi_standard"]
    assert row["exact_match_seed_sd_standard"] == 0.0
    assert row["contamination_gap_lo"] < row["contamination_gap"] < row["contamination_gap_hi"]
    assert not any(k.startswith("_") for k in row)                  # no internals leak
    agg.write_md(tmp_path / "t.md", table)
    assert "[" in (tmp_path / "t.md").read_text(encoding="utf-8")


def test_phase_suites_order_priorities_and_union():
    run_suite = _load_script("run_suite")
    p3, _ = run_suite.expand(str(ROOT / "configs" / "suite_p3_3b.yaml"))
    std = [(c.arm, c.seed) for c in p3 if c.split == "standard"]
    assert std == [("A5", 0), ("A7", 0), ("A6", 0), ("A1", 0), ("A2", 0),
                   ("A5", 1), ("A7", 1), ("A6", 1)]                # P3.1 .. P3.6 order
    assert {c.split for c in p3} == {"standard", "clean", "control"}
    p5, _ = run_suite.expand(str(ROOT / "configs" / "suite_p5_scale.yaml"))
    assert [c.priority for c in p5] == sorted((c.priority for c in p5),
                                              key=run_suite.PRIORITY_ORDER.get)
    assert {c.arm for c in p5 if c.priority == "M"} == {"A1", "A5", "A8"}
    p4, _ = run_suite.expand(str(ROOT / "configs" / "suite_p4_32b.yaml"))
    assert {(c.arm, c.model) for c in p4} == {("A3", "qwen2.5-32b"), ("A4", "qwen2.5-32b"),
                                              ("A8", "qwen2.5-3b")}
    union, suite = run_suite.expand(str(ROOT / "configs" / "suite_phase1.yaml"))
    assert suite["runs_dir"] == "runs"
    assert {c.experiment_id for c in union} == {c.experiment_id for c in p3 + p4 + p5}


def test_estimate_trains_each_adapter_once():
    from ror.config import ExperimentConfig
    from ror.estimate import estimate_plan, train_hours

    thr = {"train_tokens_per_s": {"qwen2.5-3b": 310}, "tokens_per_question": {"finqa": 1087},
           "train_questions": {"finqa": 6157}, "answers_per_s": {"qwen2.5-3b": {"answer": 1.0}},
           "eval_items": {"standard": 1147, "clean": 400}, "model_load_hours": 0.0}
    a5 = [ExperimentConfig(arm="A5", model="qwen2.5-3b", supervision="answer", split=s)
          for s in ("standard", "clean")]
    assert 6.0 < train_hours(a5[0], thr) < 6.3                   # ~6 h/epoch + selection
    est = estimate_plan(a5, thr=thr)
    assert est["adapters"] == 1 and est["evaluations"] == 2
    assert estimate_plan(a5, trained={a5[0].training_id}, thr=thr)["train"] == 0.0
