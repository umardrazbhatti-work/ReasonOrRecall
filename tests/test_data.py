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
