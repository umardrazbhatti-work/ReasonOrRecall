"""Tests for ror.report: answer analysis (pure Python) and the figure report."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.report import eval_arithmetic, match_variants, outcome  # noqa: E402


def _row(gold, pred, text, correct=False):
    return {"uid": "u", "gold": gold, "pred": pred, "text": text, "correct": correct}


def test_eval_arithmetic_only_plain_sums():
    assert eval_arithmetic("301/2575") == pytest.approx(0.116893, rel=1e-4)
    assert eval_arithmetic("286.61 - 198.09 > 0") == "yes"
    assert eval_arithmetic("(5 - 7) * 2") == -4.0
    for not_a_sum in ("-14.2%", "0.25", "", None, "__import__('os')", "x - 1", "2 ** 10"):
        assert eval_arithmetic(not_a_sum) is None


def test_outcomes_of_the_first_smoke_run():
    # real cases from the 2026-09-26 smoke run (gold is FinQA's exe_ans)
    assert outcome(_row(0.11689, 301.0, "301/2575")) == "Right arithmetic, not evaluated"
    assert outcome(_row("yes", 286.61, "286.61 - 198.09 > 0")) == "Right arithmetic, not evaluated"
    assert outcome(_row(0.14464, 0.1435, "0.1435")) == "Close (within 1%)"
    assert outcome(_row(0.24566, 137.4, "137.4/559.3")) == "Right arithmetic, not evaluated"
    assert outcome(_row("no", "yes", "yes")) == "Wrong yes/no"
    assert outcome(_row(94.0, 194.0, "194")) == "Wrong number"
    assert outcome(_row(0.145, 14.5, "14.5%")) == "Right value, written as a %"
    assert outcome(_row(0.0299, -0.0299, "-0.0299")) == "Sign flipped"
    assert outcome(_row(2.0, 200.0, "200")) == "Off by a power of 10"
    assert outcome(_row(1.0, None, "")) == "No answer found"
    assert outcome(_row(1.0, 1.0, "1", correct=True)) == "Correct"


def test_matching_rules_are_cumulative():
    rows = [(_row(0.145, 14.5, "14.5%"), "14.5%"), (_row(0.14464, 0.1435, "0.1435"), "14%"),
            (_row(0.11689, 301.0, "301/2575"), ""), (_row(0.2, 0.3, "0.3"), "20%"),
            (_row(0.14464, 0.14, "0.14"), "14%")]
    order = ["strict", "percent", "arithmetic", "within_1pct", "human_rounding"]
    for row, human in rows:
        v = match_variants(row, human)
        assert [v[k] for k in order] == sorted(v[k] for k in order)   # False... then True...
    assert match_variants(*rows[0])["percent"] and not match_variants(*rows[0])["strict"]
    assert match_variants(*rows[1])["within_1pct"]
    assert match_variants(*rows[4])["human_rounding"] and not match_variants(*rows[4])["within_1pct"]
    assert not match_variants(*rows[3])["human_rounding"]


def make_runs(root: Path) -> Path:
    """A runs folder with a trained answer arm and a program arm on both splits."""
    runs = root / "runs"
    rows = []
    specs = [("a5", "A5", "standard", 0.40, None), ("a5c", "A5", "clean", 0.30, None),
             ("a7s", "A7", "standard", 0.55, 0.9), ("a7c", "A7", "clean", 0.35, 0.8)]
    for k, (exp_id, arm, split, em, exe) in enumerate(specs):
        d = runs / exp_id
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"exp_id": exp_id, "name": exp_id,
                                                   "status": "completed"}))
        preds = [{"uid": f"q{i}", "gold": 0.1 * (i + 1), "pred": 0.1 * (i + 1) * (1 if i % 2 else 1.3),
                  "correct": i % 2 == 1, "text": str(0.1 * (i + 1)), "prompt_tokens": 800 + 40 * i,
                  "gen_tokens": 5 + i % 3} for i in range(12)]
        (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
        train = {"n_train": 100, "epochs": 2, "global_steps": 12, "train_loss": 1.2,
                 "tokens_per_s": 300.0,
                 "dev_loss": [{"step": 6, "loss": 1.1}, {"step": 12, "loss": 0.9}],
                 "selection": {"checks": [{"step": s, "dev_em": 0.2 + 0.05 * s / 3}
                                          for s in (3, 6, 9, 12)], "chosen_step": 12},
                 "train_runtime_s_this_session": 600.0, "peak_gpu_mem_gb": 6.1,
                 "log_history": [{"step": s, "loss": 2.0 - 0.1 * s, "learning_rate": 2e-4 * s / 12,
                                  "grad_norm": 1.0 + 0.1 * s} for s in range(1, 13)]
                 + [{"step": 6, "eval_loss": 1.1}, {"step": 12, "eval_loss": 0.9}]}
        rows.append({"exp_id": exp_id, "name": f"{arm}-qwen2.5-3b-finqa-{split}-s0", "arm": arm,
                     "model": "qwen2.5-3b", "split": split, "dataset": "finqa", "seed": 0,
                     "exact_match": em, "n_examples": 12, "execution_accuracy": exe,
                     "executability_rate": exe and exe + 0.05, "faithfulness_primary": exe,
                     "train_flops": 1e16, "infer_flops": 2e14 * (1 + k), "wall_time_s": 900.0,
                     "gpu": "Tesla T4", "git_commit": "abc1234", "timestamp": 1000 + k,
                     "extra": {"train": train,
                               "timing_s": {"model_and_training": 700.0, "inference": 120.0}}})
    (runs / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    failed = runs / "f1"
    failed.mkdir()
    (failed / "status.json").write_text(json.dumps({"exp_id": "f1", "status": "failed"}))
    return runs


def test_build_report_writes_figures_and_pages(tmp_path):
    pytest.importorskip("matplotlib")
    from ror.report import build_report

    runs = make_runs(tmp_path)
    files = build_report(runs)
    names = {f.name for f in files}
    for expected in ("01_scorecard.png", "02_training_curve.png",
                     "03_learning_rate_and_gradients.png", "04_answer_vs_gold.png",
                     "05_answer_outcomes.png", "06_score_by_matching_rule.png",
                     "07_error_size.png", "11_accuracy_by_prompt_length.png",
                     "12_reply_length.png", "13_time_and_compute.png",
                     "14_all_experiments.png", "15_accuracy_vs_compute.png",
                     "16_contamination_gap.png", "17_faithfulness.png",
                     "18_registry_status.png", "index.html", "report.md"):
        assert expected in names, expected
    page = (runs / "report" / "index.html").read_text(encoding="utf-8")
    assert page.count("<img ") == sum(f.suffix == ".png" for f in files)
    assert "A7-qwen2.5-3b-finqa-clean-s0" in page
    assert all(f.stat().st_size > 5000 for f in files if f.suffix == ".png")


def test_build_report_on_an_empty_folder(tmp_path):
    from ror.report import build_report

    files = build_report(tmp_path / "runs")
    assert [f.name for f in files] == ["index.html", "report.md"]


def test_report_without_matplotlib_still_writes_the_tables(tmp_path, monkeypatch):
    import ror.report as report

    def no_matplotlib():
        raise ImportError("No module named 'matplotlib'")

    monkeypatch.setattr(report, "_plt", no_matplotlib)
    files = report.build_report(make_runs(tmp_path))
    assert [f.name for f in files] == ["index.html", "report.md"]
    page = files[0].read_text(encoding="utf-8")
    assert "matplotlib is not installed" in page and "A7-qwen2.5-3b-finqa-clean-s0" in page
