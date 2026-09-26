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


def test_normalize_number_edge_forms():
    assert normalize_number(".5") == 0.5
    assert normalize_number("-.25") == -0.25
    assert normalize_number("1e-05") == 1e-05
    assert normalize_number("2.5E+3") == 2500.0
    assert normalize_number("−5") == -5.0  # unicode minus
    assert normalize_number("12.") == 12.0


def test_answers_match_ignores_trailing_period():
    assert answers_match("Yes.", "yes")


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


def test_primary_match_rule_m4():
    from ror.metrics import primary_match

    # real smoke-run answers (gold = FinQA exe_ans)
    assert primary_match(0.1435, 0.14464)                  # within 1%
    assert primary_match(0.1658, 0.16417)                  # within 1%
    assert not primary_match(0.22, 0.24566)                # 10% off
    assert primary_match(14.5, 0.145, percent_form=True)   # "14.5%" read as 0.145
    assert not primary_match(14.5, 0.145)                  # without a % sign: no
    assert not primary_match(301.0, 0.11689)               # "301/2575" is never evaluated
    assert primary_match("yes", "yes") and not primary_match("no", "yes")
    assert primary_match(0.0, 0.00005) and not primary_match(0.01, 0.0)  # strict allows ±0.001 at 0


def test_primary_accepts_everything_strict_accepts():
    from ror.metrics import answers_match, primary_match

    cases = [(0.0295, 0.02899), (15.001, 15.0), ("15%", 15.0), (1200, 1200.4), ("yes", "Yes.")]
    for pred, gold in cases:
        if answers_match(pred, gold):
            assert primary_match(pred, gold), (pred, gold)


def test_primary_exact_match_and_execution_accuracy():
    from ror.metrics import primary_exact_match, primary_execution_accuracy

    assert primary_exact_match([14.5, 0.1435, 3.0], [0.145, 0.14464, 5.0],
                               [True, False, False]) == 2 / 3
    assert primary_execution_accuracy([0.1435, None], [0.14464, 1.0]) == 0.5
