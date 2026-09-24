"""Tests for metrics and faithfulness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.metrics import normalize_number, answers_match, exact_match, execution_accuracy
from ror.faithfulness import FaithItem, program_faithfulness, executability_rate


def test_normalize_number():
    assert normalize_number("$1,200") == 1200.0
    assert normalize_number("15%") == 15.0
    assert normalize_number("(1,200)") == -1200.0
    assert normalize_number("1.2 million") == 1_200_000.0
    assert normalize_number("n/a") is None


def test_answers_match_numeric_tolerance():
    assert answers_match("15.0%", "15.0")
    assert answers_match("15.001", "15.0", rel_tol=1e-3)
    assert not answers_match("16", "15")


def test_answers_match_string():
    assert answers_match("Increase", " increase ")
    assert not answers_match("increase", "decrease")


def test_exact_match():
    assert exact_match(["15%", "20"], ["15", "20"]) == 1.0
    assert exact_match(["15", "99"], ["15", "20"]) == 0.5


def test_execution_accuracy_handles_none():
    assert execution_accuracy([15.0, None], [15.0, 20.0]) == 0.5


def test_program_faithfulness_defined_only_on_correct():
    items = [
        FaithItem(correct=True, pred_answer=15.0, exec_value=15.0),   # faithful
        FaithItem(correct=True, pred_answer=20.0, exec_value=None),    # correct but no program
        FaithItem(correct=False, pred_answer=1.0, exec_value=1.0),     # ignored (incorrect)
    ]
    # 1 of 2 correct items is faithful
    assert program_faithfulness(items) == 0.5


def test_program_faithfulness_none_when_no_correct():
    items = [FaithItem(correct=False, pred_answer=1, exec_value=1)]
    assert program_faithfulness(items) is None


def test_executability_rate():
    assert executability_rate([1.0, None, 2.0, None]) == 0.5
