"""Inference — generation, self-consistency, program execution, and the helpers
that turn predictions into faithfulness items.

IMPLEMENT the functions that raise NotImplementedError. Keep `Prediction` and the
signatures; `ror.experiment._execute` depends on them.

For PoT arms, a prediction carries the generated `program`, its executed value
`exec_value` (via ror.sandbox), and the final `answer` (the executed value, or a
parsed answer for non-PoT arms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .config import ExperimentConfig
from .data import Example
from .faithfulness import FaithItem
from .sandbox import run_program


@dataclass
class Prediction:
    uid: str
    answer: object                    # final answer used for Exact Match
    text: str = ""                    # raw model output
    program: Optional[str] = None     # PoT program if the arm emits one
    exec_value: object = None         # value from executing `program`
    samples: list = field(default_factory=list)  # self-consistency raw samples


def generate(model: Any, cfg: ExperimentConfig, examples: list[Example]) -> list[Prediction]:
    """Generate a Prediction per example.

    TODO:
      - build prompts via ror.data.build_prompt (style from cfg.supervision:
        answer/cot -> parse a number from text; pot -> execute the program)
      - batch decode; honour cfg.max_new_tokens / temperature
      - if cfg.self_consistency_k > 1: sample k times and take the majority answer
        (execute each program for PoT; majority-vote the executed values)
      - for PoT: set program + exec_value (run_program) + answer = exec_value
      - record est. inference FLOPs (ror.utils.inference_flops) for the frontier
    """
    raise NotImplementedError("implement generation + self-consistency")


def execute_program(program: str) -> tuple[Optional[object], bool]:
    """Run a PoT program in the sandbox; return (value_or_None, ok)."""
    res = run_program(program)
    return (res.value if res.ok else None), res.ok


def to_faith_items(preds: list[Prediction], golds: list) -> list[FaithItem]:
    """Build primary-faithfulness items from PoT predictions.

    A prediction is 'correct' if its final answer matches gold; the program's
    executed value is `exec_value`. (Grounding is optional — set FaithItem.grounded
    if you implement the cell-reference check.)
    """
    from .metrics import answers_match

    items = []
    for p, g in zip(preds, golds):
        items.append(FaithItem(
            correct=answers_match(p.answer, g),
            pred_answer=p.answer,
            exec_value=p.exec_value,
        ))
    return items


def posthoc_faith_items(model: Any, cfg: ExperimentConfig,
                        preds: list[Prediction], examples: list[Example]
                        ) -> Optional[list[FaithItem]]:
    """Post-hoc proxy: ask the model for a program deriving its OWN answer, run
    it, and build FaithItems (exec_value vs the model's given answer).

    Return None to skip (e.g. when it adds no signal). Report as a proxy only.

    TODO: implement the post-hoc program-induction prompt + execution. This is
    what lets RQ4 compare non-PoT arms; keep it clearly labelled as a proxy.
    """
    return None
