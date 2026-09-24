"""Shared fixtures: a tiny processed-data tree so tests never need the real data."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ror.paths import DATA_DIR_ENV  # noqa: E402

FAKE_RECORDS = [
    {"uid": "ETR/2016/page_23.pdf-2", "question": "what is the net change in net revenue?",
     "context": "| | amount |\n|---|---|\n| 2014 net revenue | $ 5735 |\n| 2015 net revenue | $ 5829 |",
     "answer": 94.0, "gold_program": "step_0 = 5829 - 5735\nanswer = step_0",
     "table_cells": [{"row": 1, "col": 1, "text": "$ 5735"}],
     "meta": {"dataset": "finqa", "program_verified": True}},
    {"uid": "X/2019/page_1.pdf-1", "question": "was revenue higher in 2019?",
     "context": "| | 2019 | 2018 |\n|---|---|---|\n| revenue | 10 | 8 |",
     "answer": "yes", "gold_program": 'step_0 = "yes" if 10 > 8 else "no"\nanswer = step_0',
     "table_cells": [], "meta": {"dataset": "finqa", "program_verified": True}},
]


def write_fake_split(data_dir: Path, dataset: str, split_file: str,
                     records=FAKE_RECORDS) -> Path:
    path = data_dir / "processed" / dataset / f"{split_file}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch) -> Path:
    """ROR_DATA_DIR pointing at a tree with finqa train/dev/test (no clean set)."""
    d = tmp_path / "data"
    for split in ("train", "dev", "test"):
        write_fake_split(d, "finqa", split)
    (d / "ror_data_manifest.json").write_text(
        json.dumps({"preprocess_version": "test", "code_commit": "abc"}), encoding="utf-8")
    monkeypatch.setenv(DATA_DIR_ENV, str(d))
    return d
