"""Inference — generation, self-consistency, program execution, and the helpers
that turn predictions into faithfulness items.

Keep `Prediction` and the signatures; `ror.experiment._execute` depends on them.

For PoT arms, a prediction carries the generated `program`, its executed value
`exec_value` (via ror.sandbox), and the final `answer` (the executed value, or a
parsed answer for non-PoT arms).

Decoding is greedy with an explicit GenerationConfig: the models' shipped
generation configs sample (Qwen2.5: temperature 0.7, top-p 0.8) and apply
repetition_penalty 1.05, which in greedy mode would still penalise digits that
already appear in the report — i.e. bias numeric answers. None of that is used.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import ExperimentConfig
from .data import PROMPT_STYLE, Example, build_messages
from .faithfulness import FaithItem
from .logging_utils import get_logger
from .metrics import normalize_number
from .sandbox import run_program

log = get_logger("ror.inference")


@dataclass
class Prediction:
    uid: str
    answer: object                    # final answer used for Exact Match
    text: str = ""                    # raw model output
    program: Optional[str] = None     # PoT program if the arm emits one
    exec_value: object = None         # value from executing `program`
    samples: list = field(default_factory=list)  # self-consistency raw samples
    prompt_tokens: int = 0            # for the compute frontier
    gen_tokens: int = 0


def generate(model: Any, cfg: ExperimentConfig, examples: list[Example]) -> list[Prediction]:
    """Generate a Prediction per example (greedy, batched, length-sorted).

    `model` is a ror.models.LoadedModel. The prompt style follows
    cfg.supervision (answer -> parse the answer from text; pot -> execute the
    program in the sandbox and use its value). Prompts are never truncated.
    """
    if cfg.self_consistency_k > 1:
        raise NotImplementedError("self-consistency (k > 1) is not implemented yet")
    if cfg.inference == "fewshot":
        # without this, A2/A4 would silently run zero-shot under a few-shot label
        raise NotImplementedError("few-shot inference (A2/A4): pass exemplars to build_messages")
    import torch
    from transformers import GenerationConfig

    style = PROMPT_STYLE[cfg.supervision]
    tok = model.tokenizer
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(build_messages(ex, style), tokenize=False,
                                       add_generation_prompt=True) for ex in examples]
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(examples)), key=lambda i: -lengths[i])  # least padding
    hf_model = model.model.get_base_model() if hasattr(model.model, "get_base_model") \
        else model.model
    gen_cfg = GenerationConfig(**decoding_settings(cfg),
                               eos_token_id=hf_model.generation_config.eos_token_id,
                               pad_token_id=tok.pad_token_id)
    # generate() merges the model's shipped defaults into any config it is given;
    # making ours the model default leaves nothing to merge in.
    hf_model.generation_config = gen_cfg
    device = next(model.model.parameters()).device
    bs = max(1, cfg.infer_batch_size)
    n_batches = (len(order) + bs - 1) // bs
    preds: list[Optional[Prediction]] = [None] * len(examples)
    t0 = time.time()
    log.info("generating %d examples (style=%s, batch=%d)", len(examples), style, bs)
    with torch.inference_mode():
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]
            batch = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
            out = model.model.generate(**batch, generation_config=gen_cfg)
            new = out[:, batch["input_ids"].shape[1]:].tolist()
            for j, i in enumerate(idx):
                ids = _strip_trailing(new[j], tok.pad_token_id)
                text = tok.decode(ids, skip_special_tokens=True).strip()
                preds[i] = _prediction(examples[i], text, style, lengths[i], len(ids))
            if (b + 1) % max(1, n_batches // 10) == 0 or b + 1 == n_batches:
                done = min((b + 1) * bs, len(order))
                rate = done / (time.time() - t0)
                log.info("  %d/%d examples (%.2f/s, ~%.0f min left)", done, len(order),
                         rate, (len(order) - done) / rate / 60 if rate else 0)
    return preds  # type: ignore[return-value]


def decoding_settings(cfg: ExperimentConfig) -> dict:
    """The decoding configuration every generation uses (recorded per result)."""
    return {"max_new_tokens": cfg.max_new_tokens, "do_sample": False,
            "repetition_penalty": 1.0}


def parse_answer(text: str) -> object:
    """Final answer from free text: 'yes'/'no', else a number (float), else the
    first line as a string; None if empty. Honours an 'answer: ...' marker."""
    t = (text or "").strip()
    marks = list(_ANSWER_MARKER.finditer(t))
    if marks:
        t = marks[-1].group(1).strip()
    line = t.splitlines()[0].strip() if t else ""
    if not line:
        return None
    yn = re.match(r"^\W*(yes|no)\b", line, re.I)
    if yn:
        return yn.group(1).lower()
    value = normalize_number(line)
    return value if value is not None else line


def extract_program(text: str) -> str:
    """The program from a model reply: a ```python block if present (even if
    unterminated), else the whole reply."""
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text or "", re.S)
    return (m.group(1) if m else text or "").strip()


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


# --- helpers ---

_ANSWER_MARKER = re.compile(r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+)", re.I | re.S)


def _strip_trailing(ids: list[int], pad_id: Optional[int]) -> list[int]:
    end = len(ids)
    while end and ids[end - 1] == pad_id:
        end -= 1
    return ids[:end]


def _prediction(ex: Example, text: str, style: str, prompt_tokens: int,
                gen_tokens: int) -> Prediction:
    if style == "pot":
        program = extract_program(text)
        value, _ = execute_program(program)
        return Prediction(uid=ex.uid, answer=value, text=text, program=program,
                          exec_value=value, prompt_tokens=prompt_tokens,
                          gen_tokens=gen_tokens)
    return Prediction(uid=ex.uid, answer=parse_answer(text), text=text,
                      prompt_tokens=prompt_tokens, gen_tokens=gen_tokens)
