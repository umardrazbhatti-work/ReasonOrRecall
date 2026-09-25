"""Tests for the data layer: loaders and preprocessing helpers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.data import (Example, dataset_fingerprint, linearize_table, load_dataset,
                      normalize_numbers, split_available)


def test_linearize_table_is_markdown_and_padded():
    out = linearize_table([["", "2019", "2018"], ["revenue", "$ 10", "8"], ["a|b", "1"]])
    assert out.splitlines() == [
        "|  | 2019 | 2018 |",
        "|---|---|---|",
        "| revenue | $ 10 | 8 |",
        "| a\\|b | 1 |  |",
    ]


def test_linearize_table_is_deterministic():
    t = [["x", " 1\n2 "], ["y", "3"]]
    assert linearize_table(t) == linearize_table([list(r) for r in t])


def test_normalize_numbers_only_strips_thousands_separators():
    assert normalize_numbers("revenue of 1,234,567 and 12,500.5") == "revenue of 1234567 and 12500.5"
    # FinQA conventions are left untouched
    assert normalize_numbers("( 41317 ) $ 5735 12 % in millions") == "( 41317 ) $ 5735 12 % in millions"
    # not thousands separators
    assert normalize_numbers("2019, 2018 and 1,23") == "2019, 2018 and 1,23"


def test_load_dataset_maps_standard_to_test(fake_data_dir):
    exs = load_dataset("finqa", "standard")
    assert len(exs) == 2 and isinstance(exs[0], Example)
    assert exs[0].answer == 94.0 and exs[1].answer == "yes"
    assert load_dataset("finqa", "train")[0].uid == exs[0].uid


def test_missing_split_is_reported_not_crashing(fake_data_dir):
    ok, why = split_available("finqa", "clean")
    assert not ok and "clean set" in why
    with pytest.raises(FileNotFoundError):
        load_dataset("finqa", "clean")
    assert split_available("finqa", "standard") == (True, "")


def test_unknown_dataset_or_split():
    with pytest.raises(ValueError):
        load_dataset("nope", "standard")
    with pytest.raises(ValueError):
        load_dataset("finqa", "validation")


def test_fingerprint_records_file_hash_and_version(fake_data_dir):
    fp = dataset_fingerprint("finqa", "standard")
    assert fp["file"].replace("\\", "/") == "processed/finqa/test.jsonl"
    assert len(fp["sha256"]) == 64 and fp["preprocess_version"] == "test"


# --- prompts and targets ---

from ror.data import build_messages, build_prompt, format_answer, format_target  # noqa: E402


def _ex(**kw) -> Example:
    base = dict(uid="u1", question="what is the change?", context="| a | b |",
                answer=0.14464, gold_program="step_0 = 1 - 2\nanswer = step_0",
                meta={"dataset": "finqa"})
    return Example(**{**base, **kw})


def test_format_answer_is_canonical():
    assert format_answer(94.0) == "94"
    assert format_answer(0.14464) == "0.14464"
    assert format_answer(-35) == "-35"
    assert format_answer(1.1197000001) == "1.1197"
    assert format_answer("yes") == "yes"


def test_format_target_answer_and_pot_gold():
    assert format_target(_ex(), "answer") == "0.14464"
    assert format_target(_ex(), "pot_gold") == "```python\nstep_0 = 1 - 2\nanswer = step_0\n```"
    with pytest.raises(ValueError):
        format_target(_ex(gold_program=None), "pot_gold")
    with pytest.raises(NotImplementedError):
        format_target(_ex(), "cot")


def test_build_prompt_has_report_question_and_instruction():
    p = build_prompt(_ex(), style="answer")
    assert p.startswith("Report:\n| a | b |")
    assert "Question: what is the change?" in p
    assert "as decimals, e.g. 0.145 for 14.5%" in p


def test_build_prompt_renders_conversation_history_and_tatqa_units():
    conv = _ex(meta={"dataset": "convfinqa",
                     "history": [{"question": "what was it in 2007?", "answer": 60.94}]})
    assert "Conversation so far:\nQ: what was it in 2007?\nA: 60.94" in build_prompt(conv, style="answer")
    tat = build_prompt(_ex(meta={"dataset": "tatqa"}), style="answer")
    assert "percent values" in tat and "as decimals" not in tat


def test_fewshot_exemplars_show_targets_like_training():
    p = build_prompt(_ex(uid="q"), fewshot=[_ex(uid="e1", answer=94.0)], style="answer")
    assert "Example 1" in p and "Answer:\n94" in p and p.count("Report:") == 2


def test_build_messages_is_system_plus_user():
    msgs = build_messages(_ex(), "pot")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "```python" in msgs[1]["content"]
