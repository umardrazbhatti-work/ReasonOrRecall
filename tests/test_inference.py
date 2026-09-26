"""Tests for answer parsing and program extraction (no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.inference import _strip_trailing, execute_program, extract_program, parse_answer


def test_parse_answer_plain_numbers_and_yes_no():
    assert parse_answer("0.14464") == 0.14464
    assert parse_answer("-35") == -35.0
    assert parse_answer("Yes") == "yes"
    assert parse_answer("no.") == "no"
    assert parse_answer("") is None


def test_parse_answer_uses_the_last_answer_marker():
    assert parse_answer("First 2019 revenue was 10.\nThe answer is 0.25") == 0.25
    assert parse_answer("Final answer: $1,200") == 1200.0


def test_parse_answer_keeps_text_spans():
    assert parse_answer("Rate of inflation; Discount rate") == "Rate of inflation; Discount rate"
    assert parse_answer("nothing numeric") == "nothing numeric"


def test_extract_program_from_fences():
    assert extract_program("```python\nanswer = 1\n```") == "answer = 1"
    assert extract_program("Here:\n```\nx = 2\nanswer = x\n```\nDone") == "x = 2\nanswer = x"
    assert extract_program("```python\nanswer = 3\n") == "answer = 3"   # unterminated
    assert extract_program("answer = 4") == "answer = 4"


def test_extracted_program_executes_in_sandbox():
    value, ok = execute_program(extract_program("```python\nstep_0 = 5829 - 5735\nanswer = step_0\n```"))
    assert ok and value == 94


def test_strip_trailing_padding_only():
    assert _strip_trailing([5, 6, 0, 0], 0) == [5, 6]
    assert _strip_trailing([0, 5, 0], 0) == [0, 5]


def test_unimplemented_decoding_modes_refuse_instead_of_mislabelling():
    import pytest

    from ror.config import ExperimentConfig
    from ror.inference import generate

    fewshot = ExperimentConfig(arm="A2", model="qwen2.5-3b", supervision="none",
                               inference="fewshot")
    with pytest.raises(NotImplementedError, match="few-shot"):
        generate(model=None, cfg=fewshot, examples=[])     # before any model is touched
    with pytest.raises(NotImplementedError, match="self-consistency"):
        sc = ExperimentConfig(arm="A5", model="qwen2.5-3b", self_consistency_k=5)
        generate(model=None, cfg=sc, examples=[])


def test_answer_is_percent_reads_the_final_answer():
    from ror.inference import answer_is_percent

    assert answer_is_percent("-14.2%")
    assert answer_is_percent("The margin grew.\nAnswer: 14.5 %")
    assert not answer_is_percent("0.145")
    assert not answer_is_percent("Growth was 14%.\nAnswer: 0.14")
