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
  "control"  -> clean_set/control.jsonl: the same templates on FinQA test
                tables (pre-cutoff), separating question style from recency

Split naming used across the repo:
  "train"    -> training split
  "dev"      -> development split (validation / early stopping)
  "standard" -> the benchmark evaluation split (the potentially-contaminated one)
  "clean"    -> the post-cutoff held-out set
  "control"  -> the style-control set (templated questions, pre-cutoff tables)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .paths import clean_set_path, control_set_path, data_dir, manifest_path, processed_path

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

    name in {finqa, convfinqa, tatqa}; split in {train, dev, standard, clean,
    control} ("test" is accepted as an alias of the raw test file). "clean" and
    "control" load clean_set/{clean,control}.jsonl regardless of `name`.
    """
    path = split_path(name, split)
    if not path.exists():
        raise FileNotFoundError(_missing_message(name, split, path))
    return [example_from_record(r) for r in _read_jsonl(path)]


def split_path(name: str, split: str) -> Path:
    """Where `load_dataset(name, split)` reads from."""
    if split == "clean":
        return clean_set_path()
    if split == "control":
        return control_set_path()
    if name not in STANDARD_SPLIT:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    files = {"train": "train", "dev": "dev", "test": "test",
             "standard": STANDARD_SPLIT[name]}
    if split not in files:
        raise ValueError(f"unknown split {split!r}; known: {', '.join(files)}, clean, "
                         f"control")
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
    info = {"file": path.relative_to(data_dir()).as_posix() if path.is_relative_to(data_dir())
            else path.as_posix(), "sha256": _sha256(path)}
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


# --- prompts and targets -------------------------------------------------------
# Training (ror.training) and inference (ror.inference) both build their input
# with `build_messages`, so a model is evaluated on exactly the prompt format it
# was trained on. Changing any text below changes model inputs: treat it as part
# of the method.

SYSTEM_PROMPT = ("You are a financial analyst. Answer the question using only the "
                 "report excerpt provided (its text and table).")

_INSTRUCTIONS = {
    ("answer", "default"): (
        "Give only the final answer: a number (write ratios and percentages as "
        "decimals, e.g. 0.145 for 14.5%) or yes/no."),
    ("answer", "tatqa"): (
        "Give only the final answer: a number in the report's units (percentages "
        "as percent values, e.g. 14.5), a text span, or several spans separated "
        "by '; '."),
    ("pot", "default"): (
        "Write a Python program that computes the answer from the numbers in the "
        "report and assigns it to a variable named `answer` (ratios and "
        "percentages as decimals; \"yes\"/\"no\" for comparisons). Reply with only "
        "the program in a ```python code block."),
    ("pot", "tatqa"): (
        "Write a Python program that computes the answer from the numbers in the "
        "report and assigns it to a variable named `answer`, in the report's units "
        "(percentages as percent values). Reply with only the program in a "
        "```python code block."),
}

# which prompt style each supervision format trains/evaluates with
PROMPT_STYLE = {"none": "answer", "answer": "answer", "cot": "cot",
                "pot_gold": "pot", "pot_distilled": "pot"}
_SUPERVISION_FOR_STYLE = {"answer": "answer", "cot": "cot", "pot": "pot_gold"}


def format_answer(value: object) -> str:
    """Canonical text of a gold answer: integers without a decimal point, other
    numbers to at most 5 decimals (FinQA's execution precision), strings as-is."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.5f}".rstrip("0").rstrip(".")


def format_target(example: Example, supervision: str) -> str:
    """Build the supervised target string for a training example.

    supervision:
      "answer"        -> just the final answer (`format_answer`)
      "cot"           -> a natural-language rationale then the answer (A6; TODO)
      "pot_gold"      -> the gold executable program in a ```python block
                         (execution-verified in ror.preprocess)
      "pot_distilled" -> teacher programs come from training.load_distilled_traces
    """
    if supervision == "answer":
        return format_answer(example.answer)
    if supervision == "pot_gold":
        if not example.gold_program:
            raise ValueError(f"{example.uid} has no verified gold program")
        return f"```python\n{example.gold_program}\n```"
    if supervision == "cot":
        raise NotImplementedError("CoT targets (A6): design the rationale format first")
    if supervision == "pot_distilled":
        raise ValueError("pot_distilled targets are teacher traces "
                         "(training.load_distilled_traces), not format_target")
    raise ValueError(f"unknown supervision {supervision!r}")


def build_prompt(example: Example, fewshot: list[Example] | None = None,
                 style: str = "pot") -> str:
    """Assemble the user message (optionally with few-shot exemplars).

    style in {"answer", "cot", "pot"} controls what the model is asked to emit.
    ConvFinQA items carry the earlier turns in example.meta["history"]. Exemplars
    show their targets exactly as `format_target` would produce them.
    """
    if style not in ("answer", "pot"):
        raise NotImplementedError(f"prompt style {style!r} (CoT is A6; not designed yet)")
    parts = []
    for i, ex in enumerate(fewshot or [], 1):
        target = format_target(ex, _SUPERVISION_FOR_STYLE[style])
        parts.append(f"Example {i}\n\n{_task_block(ex)}\n\nAnswer:\n{target}")
    if parts:
        parts.append("Now answer this one.")
    parts.append(_task_block(example))
    dataset = "tatqa" if example.meta.get("dataset") == "tatqa" else "default"
    parts.append(_INSTRUCTIONS[(style, dataset)])
    return "\n\n".join(parts)


def build_messages(example: Example, style: str,
                   fewshot: list[Example] | None = None) -> list[dict]:
    """Chat messages (system + user) for training and inference alike."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(example, fewshot, style)}]


def _task_block(example: Example) -> str:
    history = example.meta.get("history") or []
    convo = ""
    if history:
        turns = "\n".join(f"Q: {h['question']}\nA: {format_answer(h['answer'])}"
                          for h in history)
        convo = f"Conversation so far:\n{turns}\n\n"
    return f"Report:\n{example.context}\n\n{convo}Question: {example.question}"


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
    if split in ("clean", "control"):
        return (f"{split} set not built yet ({path}); run scripts/build_clean_set.py "
                f"and add it to the data directory")
    return (f"{name}/{split} data not found at {path}; run scripts/prepare_data.py, "
            f"or on Kaggle attach the ror-data dataset (ROR_DATA_DIR)")
