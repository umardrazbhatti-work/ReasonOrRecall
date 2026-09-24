"""Data layer — loaders, preprocessing, and supervision-target formatting.

IMPLEMENT the functions that raise NotImplementedError. Keep the `Example`
dataclass and the public signatures exactly as given — `ror.experiment` and the
metrics depend on them.

Datasets (see proposal 5.1):
  finqa      -> ibm-research/finqa  (splits: train / dev / test == "standard")
  convfinqa  -> multi-turn; evaluate on the labelled dev split
  tatqa      -> hybrid table+text
  "clean"    -> the contamination-controlled set built by
                scripts/build_clean_set.py, loaded from data/clean_set/

Split naming used across the repo:
  "train"    -> training split
  "standard" -> the benchmark test/dev split (the potentially-contaminated one)
  "clean"    -> the post-cutoff held-out set
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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

    name in {finqa, convfinqa, tatqa}; split in {train, standard, clean}.
    For "clean", load from data/clean_set/ regardless of `name`.

    TODO: use `datasets.load_dataset`, map raw fields into Example, and call
    `linearize_table` + `normalize_numbers` in preprocessing. Confirm the dataset
    license before committing any cached copy.
    """
    raise NotImplementedError("implement dataset loading — see docstring")


def linearize_table(table: list[list[str]]) -> str:
    """Turn a 2-D table into text with explicit row/column markers.

    TODO: a simple, deterministic serialization (e.g. markdown-ish rows with a
    header). Keep it stable — it is part of the model input.
    """
    raise NotImplementedError("implement table linearization")


def normalize_numbers(text: str) -> str:
    """Normalize numeric surface forms in free text if needed (scale words,
    parentheses-negatives, separators). Number *parsing* for scoring lives in
    ror.metrics.normalize_number; this is for cleaning model input if useful."""
    raise NotImplementedError("implement or return text unchanged")


def format_target(example: Example, supervision: str) -> str:
    """Build the supervised target string for a training example.

    supervision:
      "answer"        -> just the final answer
      "cot"           -> a natural-language rationale then the answer
      "pot_gold"      -> the gold executable program (must assign `answer`)
      "pot_distilled" -> a teacher-generated program (provided at train time;
                         this function may be unused for that arm)

    TODO: implement the answer/cot/pot_gold cases. See proposal 5.3 for the
    exact three-format scheme and the worked example.
    """
    raise NotImplementedError("implement target formatting per supervision")


def build_prompt(example: Example, fewshot: list[Example] | None = None,
                 style: str = "pot") -> str:
    """Assemble the inference prompt (optionally with few-shot exemplars).

    style in {"answer", "cot", "pot"} controls what the model is asked to emit.
    TODO: implement; keep exemplar formatting consistent with `format_target`.
    """
    raise NotImplementedError("implement prompt construction")
