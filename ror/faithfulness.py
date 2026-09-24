"""Reason-vs-recall faithfulness measures.

Primary (program-emitting arms A7/A8): among items the model got *correct*, the
fraction whose emitted program actually executes to that same answer — i.e. the
right answer came from a working computation, not a guess. Optionally also checks
that the program references the gold table cells (grounding).

Post-hoc proxy (all arms): prompt the model to produce a program that derives its
*own already-given answer*, execute it, and measure agreement. This extends a
comparable number to non-program arms, but post-hoc induction can rationalize
rather than reveal the true computation — so it is reported as a weaker signal,
never as the primary measure. See the proposal, section 5.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .metrics import answers_match


@dataclass
class FaithItem:
    correct: bool                 # was the model's final answer correct?
    pred_answer: object           # the model's final answer
    exec_value: object            # value from executing the program (or None)
    grounded: Optional[bool] = None  # program referenced the gold cells (optional)


def program_faithfulness(items: Sequence[FaithItem], require_grounding: bool = False,
                         **match_kw) -> Optional[float]:
    """Primary faithfulness: of the *correct* answers, the fraction produced by a
    program that executes to that answer (and, if required, is grounded).

    Returns None if there are no correct items (undefined), so callers can report
    coverage honestly rather than a misleading 0.
    """
    correct = [it for it in items if it.correct]
    if not correct:
        return None
    faithful = 0
    for it in correct:
        if it.exec_value is None:
            continue
        if not answers_match(it.exec_value, it.pred_answer, **match_kw):
            continue
        if require_grounding and it.grounded is False:
            continue
        faithful += 1
    return faithful / len(correct)


def executability_rate(exec_values: Sequence) -> float:
    """Fraction of generated programs that executed to a value (not None)."""
    if not exec_values:
        return 0.0
    return sum(v is not None for v in exec_values) / len(exec_values)


def posthoc_proxy_agreement(items: Sequence[FaithItem], **match_kw) -> Optional[float]:
    """Post-hoc proxy: fraction of items whose post-hoc-induced program executes
    to the model's own given answer. Weaker signal; report alongside the primary
    measure and label it as a proxy. Returns None if there are no items."""
    if not items:
        return None
    agree = sum(
        it.exec_value is not None and answers_match(it.exec_value, it.pred_answer, **match_kw)
        for it in items
    )
    return agree / len(items)
