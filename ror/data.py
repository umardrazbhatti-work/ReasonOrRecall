"""Data layer — loaders, preprocessing helpers, and supervision-target formatting.

Keep the `Example` dataclass and the public signatures exactly as given —
`ror.experiment` and the metrics depend on them.

Datasets (proposal 5.1). Raw sources are pinned to exact upstream commits and
turned into Example-shaped JSONL by `scripts/prepare_data.py` (logic in
`ror.preprocess`). Loading reads that JSONL from the data directory
($ROR_DATA_DIR, default <repo>/data — on Kaggle, the attached dataset):

  finqa      -> processed/finqa/{train,dev,test}.jsonl         standard = test
  convfinqa  -> processed/convfinqa/{train,dev}.jsonl (turns)  standard = dev
                (test answers are private; the labelled dev split is the anchor)
  tatqa      -> processed/tatqa/{train,dev,test}.jsonl         standard = test (gold)
  "clean"    -> clean_set/clean.jsonl, built by scripts/build_clean_set.py

Split naming used across the repo:
  "train"    -> training split
  "dev"      -> development split (validation / early stopping)
  "standard" -> the benchmark evaluation split (the potentially-contaminated one)
  "clean"    -> the post-cutoff held-out set
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .paths import clean_set_path, data_dir, manifest_path, processed_path

# the benchmark split each dataset is evaluated on as "standard"
STANDARD_SPLIT = {"finqa": "test", "convfinqa": "dev", "tatqa": "test"}
DATASETS = tuple(STANDARD_SPLIT)


@dataclass
class Example:
    """One QA item. `context` is the linearized table + surrounding text."""
    uid: str
    question: str
    context: str
    answer: object                      # gold answer (number or string)
    gold_program: Optional[str] = None   # gold PoT program if available
    table_cells: list = field(default_factory=list)  # for grounding checks
    meta: dict = field(default_factory=dict)


def load_dataset(name: str, split: str) -> list[Example]:
    """Load a dataset split as a list[Example].

    name in {finqa, convfinqa, tatqa}; split in {train, dev, standard, clean}
    ("test" is accepted as an alias of the raw test file). For "clean", load
    clean_set/clean.jsonl regardless of `name`.
    """
    path = split_path(name, split)
    if not path.exists():
        raise FileNotFoundError(_missing_message(name, split, path))
    return [example_from_record(r) for r in _read_jsonl(path)]


def split_path(name: str, split: str) -> Path:
    """Where `load_dataset(name, split)` reads from."""
    if split == "clean":
        return clean_set_path()
    if name not in STANDARD_SPLIT:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    files = {"train": "train", "dev": "dev", "test": "test",
             "standard": STANDARD_SPLIT[name]}
    if split not in files:
        raise ValueError(f"unknown split {split!r}; known: {', '.join(files)}, clean")
    return processed_path(name, files[split])


def split_available(name: str, split: str) -> tuple[bool, str]:
    """(True, "") if the split's data is present, else (False, reason). Used to
    skip experiments whose data is missing *before* they consume an attempt."""
    try:
        path = split_path(name, split)
    except ValueError as e:
        return False, str(e)
    if path.exists():
        return True, ""
    return False, _missing_message(name, split, path)


def dataset_fingerprint(name: str, split: str) -> dict:
    """Provenance for a result row: which file, its sha256, and the data manifest
    version it came from."""
    path = split_path(name, split)
    info = {"file": str(path.relative_to(data_dir())) if path.is_relative_to(data_dir())
            else str(path), "sha256": _sha256(path)}
    if manifest_path().exists():
        m = json.loads(manifest_path().read_text(encoding="utf-8"))
        info["preprocess_version"] = m.get("preprocess_version")
        info["data_code_commit"] = m.get("code_commit")
    return info


def example_from_record(record: dict) -> Example:
    known = Example.__dataclass_fields__  # type: ignore[attr-defined]
    return Example(**{k: v for k, v in record.items() if k in known})


def example_to_record(example: Example) -> dict:
    return asdict(example)


def linearize_table(table: list[list[str]]) -> str:
    """Turn a 2-D table into a markdown table (first row = header).

    Deterministic and stable — it is part of the model input. Ragged rows are
    padded, pipes inside cells are escaped, and newlines are flattened.
    """
    rows = [[_clean_cell(c) for c in row] for row in table if row is not None]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _clean_cell(cell: object) -> str:
    return re.sub(r"\s+", " ", str(cell if cell is not None else "")).strip().replace("|", "\\|")


_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_numbers(text: str) -> str:
    """Normalize numeric surface forms in model input.

    Only thousands separators are removed ("1,234,567" -> "1234567"), so filings
    with commas (TAT-QA, the EDGAR clean set) match FinQA's comma-free style.
    Deliberately left alone:
      - parenthesised numbers "( 41317 )": in FinQA's gold programs they are used
        as positive magnitudes (11/11 train programs that reference one), so
        rewriting them as negatives would contradict the gold;
      - scale words ("in millions"): gold answers are in the table's units;
      - "$ 5735" / "12 %" spacing: FinQA's native tokenisation.
    Number *parsing* for scoring lives in ror.metrics.normalize_number.
    """
    return _THOUSANDS_SEP.sub("", text)


def format_target(example: Example, supervision: str) -> str:
    """Build the supervised target string for a training example.

    supervision:
      "answer"        -> just the final answer
      "cot"           -> a natural-language rationale then the answer
      "pot_gold"      -> the gold executable program (must assign `answer`)
      "pot_distilled" -> a teacher-generated program (provided at train time;
                         this function may be unused for that arm)

    TODO: implement the answer/cot/pot_gold cases. See proposal 5.3 for the
    exact three-format scheme and the worked example. (`gold_program` is already
    execution-verified Python for FinQA/ConvFinQA, see ror.preprocess.)
    """
    raise NotImplementedError("implement target formatting per supervision")


def build_prompt(example: Example, fewshot: list[Example] | None = None,
                 style: str = "pot") -> str:
    """Assemble the inference prompt (optionally with few-shot exemplars).

    style in {"answer", "cot", "pot"} controls what the model is asked to emit.
    ConvFinQA items carry the earlier turns in example.meta["history"].
    TODO: implement; keep exemplar formatting consistent with `format_target`.
    """
    raise NotImplementedError("implement prompt construction")


# --- helpers ---

def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _missing_message(name: str, split: str, path: Path) -> str:
    if split == "clean":
        return (f"clean set not built yet ({path}); run scripts/build_clean_set.py "
                f"and add it to the data directory")
    return (f"{name}/{split} data not found at {path}; run scripts/prepare_data.py, "
            f"or on Kaggle attach the ror-data dataset (ROR_DATA_DIR)")
