"""Tests for raw -> Example preprocessing and gold-program conversion."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.preprocess import (ProgramError, build_finqa, execute_trusted,
                            finqa_program_to_python, grounding_cells,
                            tatqa_derivation_to_expr, verified_program)
from ror.sandbox import run_program

TABLE = [["", "2016", "2015"],
         ["net revenue", "$ 5829", "$ 5735"],
         ["margin", "22% ( 22 % )", "26% ( 26 % )"]]


@pytest.mark.parametrize("dsl,expected", [
    ("subtract(5829, 5735)", 94),
    ("subtract(5829, 5735), divide(#0, 5735)", (5829 - 5735) / 5735),
    ("add(const_1, 2.0%), exp(#0, 7)", 1.02 ** 7),
    ("multiply(3.3, const_1000000)", 3.3e6),
    ("subtract(const_m1, 4)", -5),
    ("greater(5829, 5735)", "yes"),
    ("table_average(margin, none)", 0.24),
    ("table_max(net revenue, none)", 5829),
    ("25.14", 25.14),  # ConvFinQA lookup turn
])
def test_finqa_program_conversion_executes_to_expected(dsl, expected):
    code = finqa_program_to_python(dsl, TABLE)
    assert "answer =" in code
    value = execute_trusted(code)
    if isinstance(expected, str):
        assert value == expected
    else:
        assert abs(value - expected) < 1e-9


def test_converted_program_runs_identically_in_sandbox():
    code = finqa_program_to_python("subtract(5829, 5735), divide(#0, 5735)", TABLE)
    res = run_program(code)
    assert res.ok and abs(res.value - execute_trusted(code)) < 1e-12


def test_negative_literals_are_parenthesised():
    assert execute_trusted(finqa_program_to_python("exp(-3, 2)")) == 9


@pytest.mark.parametrize("bad", ["", "subtract(1, 2", "frobnicate(1, 2)",
                                 "divide(#1, 2)", "table_sum(no such row, none)"])
def test_malformed_programs_raise(bad):
    with pytest.raises(ProgramError):
        finqa_program_to_python(bad, TABLE)


def test_verification_rejects_wrong_programs():
    code = finqa_program_to_python("subtract(5829, 5735)")
    assert verified_program(code, 94.0)
    assert not verified_program(code, 95.0)


def test_tatqa_derivations():
    assert tatqa_derivation_to_expr("(1,617-1,434)/1,434") == "(1617-1434)/1434"
    assert tatqa_derivation_to_expr("[(2.9+2.9)/2] - [(2.7+2.7)/2]") == "((2.9+2.9)/2) - ((2.7+2.7)/2)"
    assert tatqa_derivation_to_expr("12.5% * 200") == "(12.5/100) * 200"
    with pytest.raises(ProgramError):
        tatqa_derivation_to_expr("__import__('os')")


def test_grounding_cells_find_operands_and_table_rows():
    cells = grounding_cells(TABLE, "subtract(5829, 5735)")
    assert {(c["row"], c["col"]) for c in cells} == {(1, 1), (1, 2)}
    cells = grounding_cells(TABLE, "table_average(margin, none)")
    assert {(c["row"], c["col"]) for c in cells} == {(2, 1), (2, 2)}


def test_build_finqa_record_shape(tmp_path):
    raw = {"id": "ETR/2016/page_23.pdf-2", "filename": "ETR/2016/page_23.pdf",
           "pre_text": ["net revenue 1,000 rose ."], "post_text": ["end ."],
           "table": TABLE,
           "qa": {"question": "what is the change?", "answer": "94",
                  "program": "subtract(5829, 5735)", "exe_ans": 94.0,
                  "gold_inds": {"table_1": "..."}}}
    (tmp_path / "finqa").mkdir()
    for split in ("train", "dev", "test"):
        (tmp_path / "finqa" / f"{split}.json").write_text(json.dumps([raw]), encoding="utf-8")
    rec = build_finqa(tmp_path)["test"][0]
    assert set(rec) == {"uid", "question", "context", "answer", "gold_program",
                        "table_cells", "meta"}
    assert rec["answer"] == 94.0 and rec["meta"]["program_verified"]
    assert rec["gold_program"] == "step_0 = 5829 - 5735\nanswer = step_0"
    assert "net revenue 1000 rose" in rec["context"]      # thousands separator removed
    assert "| net revenue | $ 5829 | $ 5735 |" in rec["context"]
