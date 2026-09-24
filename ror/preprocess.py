"""Raw benchmark files -> Example-shaped records (the `processed/` JSONL).

One concern: turning pinned upstream files into the repo's `Example` schema.
`scripts/prepare_data.py` orchestrates download -> build -> write -> zip; the
loaders in `ror.data` read the result.

Gold answers and programs
  - FinQA / ConvFinQA: `answer` is the executed value `exe_ans` (ratios as
    decimals, e.g. 0.14464 for "14%"; "yes"/"no" for comparisons) — the value
    FinQA's own execution accuracy is scored against, and always present (the
    annotator's `answer` string is sometimes empty; it is kept in meta).
    The FinQA DSL program is converted to Python that assigns `answer`
    (`finqa_program_to_python`) and kept as `gold_program` only if executing it
    reproduces `exe_ans`.
  - TAT-QA: `answer` is the gold answer in the table's units (see meta.scale);
    arithmetic items get a Python `gold_program` from `derivation` when it
    executes to the gold answer.

Only programs *generated here* from numeric literals and a fixed set of
operators are executed in-process (`execute_trusted`); model output always goes
through ror.sandbox.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .data import linearize_table, normalize_numbers
from .metrics import answers_match

# Bump when the processed output changes for the same raw input.
PREPROCESS_VERSION = "1"


@dataclass(frozen=True)
class Source:
    """A pinned upstream dataset: files are fetched from an exact commit."""
    name: str
    repo: str
    commit: str
    license: str
    files: dict = field(default_factory=dict)   # local name -> path in repo
    citation: str = ""

    def url(self, repo_path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{repo_path}"


SOURCES: dict[str, Source] = {
    "finqa": Source(
        name="finqa", repo="czyssrs/FinQA",
        commit="0f16e2867befa6840783e58be38c9efb9229d742", license="MIT",
        files={"train.json": "dataset/train.json", "dev.json": "dataset/dev.json",
               "test.json": "dataset/test.json"},
        citation="Chen et al. FinQA: A Dataset of Numerical Reasoning over Financial "
                 "Data. EMNLP 2021. arXiv:2109.00122"),
    "convfinqa": Source(
        name="convfinqa", repo="czyssrs/ConvFinQA",
        commit="cf3eed2d5984960bf06bb8145bcea5e80b0222a6", license="MIT",
        files={"data.zip": "data.zip"},
        citation="Chen et al. ConvFinQA: Exploring the Chain of Numerical Reasoning in "
                 "Conversational Finance Question Answering. EMNLP 2022. arXiv:2210.03849"),
    "tatqa": Source(
        name="tatqa", repo="NExTplusplus/TAT-QA",
        commit="870accc41953dcde885aabeb963d94aabdc0fbc3",
        license="CC BY 4.0 (dataset; repository code MIT)",
        files={"train.json": "dataset_raw/tatqa_dataset_train.json",
               "dev.json": "dataset_raw/tatqa_dataset_dev.json",
               "test_gold.json": "dataset_raw/tatqa_dataset_test_gold.json"},
        citation="Zhu et al. TAT-QA: A Question Answering Benchmark on a Hybrid of "
                 "Tabular and Textual Content in Finance. ACL 2021"),
}


# --- download ---

def download_source(src: Source, raw_dir: Path, force: bool = False,
                    log: Callable[[str], None] = print) -> dict[str, str]:
    """Fetch every file of `src` into raw_dir/<name>/; return {file: sha256}."""
    out = raw_dir / src.name
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for local, repo_path in src.files.items():
        dest = out / local
        if force or not dest.exists():
            log(f"download {src.url(repo_path)} -> {dest}")
            req = urllib.request.Request(src.url(repo_path),
                                         headers={"User-Agent": "reason-or-recall"})
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        hashes[local] = sha256_file(dest)
    return hashes


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- FinQA DSL -> Python ---

class ProgramError(ValueError):
    """A gold program that cannot be converted (malformed or unsupported)."""


_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**"}
_TABLE_OPS = {"table_max", "table_min", "table_sum", "table_average"}


def _fmt_num(v: float) -> str:
    """Compact Python literal; negatives parenthesised so `a ** -b` etc. stay exact."""
    s = str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))
    return f"({s})" if s.startswith("-") else s


def _str_to_num(text: str) -> Optional[float]:
    """FinQA's str_to_num: commas dropped, 'x%' -> x/100, const_N / const_m1."""
    t = text.replace(",", "").strip()
    try:
        return float(t)
    except ValueError:
        pass
    if "%" in t:
        try:
            return float(t.replace("%", "")) / 100.0
        except ValueError:
            return None
    if t.startswith("const_"):
        c = t[len("const_"):]
        try:
            return -1.0 if c == "m1" else float(c)
        except ValueError:
            return None
    return None


def _arg_literal(arg: str, step: int) -> str:
    a = arg.strip()
    if a.startswith("#"):
        try:
            ref = int(a[1:])
        except ValueError:
            raise ProgramError(f"bad step reference {a!r}") from None
        if not 0 <= ref < step:
            raise ProgramError(f"reference {a!r} to a later step")
        return f"step_{ref}"
    t = a.replace(",", "")
    if _PLAIN_NUMBER.fullmatch(t):  # keep the source's own digits (grounding)
        return f"({t})" if t.startswith("-") else t
    if t.endswith("%") and _PLAIN_NUMBER.fullmatch(t[:-1].strip()):
        return f"({t[:-1].strip()} / 100)"
    v = _str_to_num(a)
    if v is None:
        raise ProgramError(f"unparseable argument {a!r}")
    return _fmt_num(v)


def _tokenize(program: str) -> list[str]:
    """FinQA's own program tokenisation (split on ', ', then on brackets)."""
    toks: list[str] = []
    for tok in program.split(", "):
        cur = ""
        for c in tok:
            if c == ")":
                if cur:
                    toks.append(cur)
                    cur = ""
            cur += c
            if c in "()":
                toks.append(cur)
                cur = ""
        if cur:
            toks.append(cur)
    return toks


def table_row_values(table: list[list[str]], row_name: str) -> list[float]:
    """FinQA table-op semantics: find the row by its first cell (the last row with
    that name wins), parse every other cell ($ stripped, text after '(' dropped)."""
    rows = {str(r[0]).strip(): r[1:] for r in table if r}
    name = row_name.strip()
    if name not in rows:
        raise ProgramError(f"table row {name!r} not found")
    vals = []
    for cell in rows[name]:
        c = str(cell).replace("$", "").strip().split("(")[0].strip()
        v = _str_to_num(c)
        if v is None:
            raise ProgramError(f"non-numeric cell {cell!r} in row {name!r}")
        vals.append(v)
    if not vals:
        raise ProgramError(f"table row {name!r} is empty")
    return vals


def finqa_program_to_python(program: str, table: Optional[list[list[str]]] = None) -> str:
    """Convert a FinQA/ConvFinQA DSL program to Python that assigns `answer`.

        subtract(5829, 5735), divide(#0, 5735)
    ->  step_0 = 5829 - 5735
        step_1 = step_0 / 5735
        answer = step_1

    A bare number (ConvFinQA lookup turns) becomes `answer = <number>`.
    Raises ProgramError for malformed or unsupported programs.
    """
    program = (program or "").strip()
    if not program:
        raise ProgramError("empty program")
    if "(" not in program:
        return f"answer = {_arg_literal(program, 0)}"
    toks = _tokenize(program)
    if len(toks) % 4:
        raise ProgramError(f"malformed program {program!r}")
    lines = []
    for step in range(len(toks) // 4):
        op_tok, a1, a2, close = toks[4 * step: 4 * step + 4]
        if not op_tok.endswith("(") or close != ")":
            raise ProgramError(f"malformed step {step} in {program!r}")
        op = op_tok[:-1].strip()
        if op in _OPS:
            expr = f"{_arg_literal(a1, step)} {_OPS[op]} {_arg_literal(a2, step)}"
        elif op == "greater":
            expr = f'"yes" if {_arg_literal(a1, step)} > {_arg_literal(a2, step)} else "no"'
        elif op in _TABLE_OPS:
            if table is None:
                raise ProgramError("table op without a table")
            row = table_row_values(table, a1)
            vals = "[" + ", ".join(_fmt_num(v) for v in row) + "]"
            expr = {"table_max": f"max({vals})", "table_min": f"min({vals})",
                    "table_sum": f"sum({vals})",
                    "table_average": f"sum({vals}) / {len(row)}"}[op]
        else:
            raise ProgramError(f"unknown op {op!r}")
        lines.append(f"step_{step} = {expr}")
    lines.append(f"answer = step_{len(toks) // 4 - 1}")
    return "\n".join(lines)


# --- TAT-QA derivation -> Python ---

_DERIV_ALLOWED = re.compile(r"[0-9.+\-*/() ]+")


def tatqa_derivation_to_expr(derivation: str) -> str:
    """Turn a TAT-QA arithmetic derivation into a Python expression.

    "(1,617-1,434)/1,434" -> "(1617-1434)/1434";  "[a] - [b]" brackets become
    parentheses; "12.5%" -> "(12.5/100)". Anything else raises ProgramError.
    """
    e = (derivation or "").strip()
    if not e:
        raise ProgramError("empty derivation")
    e = (e.replace("[", "(").replace("]", ")").replace("×", "*")
          .replace("÷", "/").replace("−", "-").replace("$", ""))
    e = normalize_numbers(e)
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)
    e = re.sub(r"\s+", " ", e).strip()
    if not _DERIV_ALLOWED.fullmatch(e):
        raise ProgramError(f"unsupported derivation {derivation!r}")
    return e


# --- trusted execution + verification ---

_SAFE_BUILTINS = {"max": max, "min": min, "sum": sum}


def execute_trusted(code: str) -> object:
    """Execute a program generated by this module (numeric literals and fixed
    operators only) in-process. NEVER use for model output — use ror.sandbox."""
    ns: dict = {}
    exec(compile(code, "<gold_program>", "exec"), {"__builtins__": _SAFE_BUILTINS}, ns)
    return ns["answer"]


def gold_matches(value: object, gold: object, round_2dp: bool = False) -> bool:
    """Gold-program verification rule (metrics tolerance; TAT-QA gold is rounded to 2 dp)."""
    if answers_match(value, gold):
        return True
    if round_2dp and isinstance(value, (int, float)) and isinstance(gold, (int, float)):
        return abs(value - gold) <= 0.005 + 1e-9 * abs(gold)  # TAT-QA rounds to 2 dp
    return False


def verified_program(code: str, gold: object, round_2dp: bool = False) -> bool:
    try:
        return gold_matches(execute_trusted(code), gold, round_2dp)
    except Exception:  # noqa: BLE001 — any failure means "not verified"
        return False


# --- shared context building ---

def normalize_table(table: list[list[str]]) -> list[list[str]]:
    return [[normalize_numbers(str(c)) for c in row] for row in table]


def build_context(pre_text: list[str], table: list[list[str]],
                  post_text: list[str]) -> str:
    """pre-text, the linearized table, post-text — blank-line separated."""
    parts = [normalize_numbers(" ".join(pre_text)).strip(),
             linearize_table(normalize_table(table)),
             normalize_numbers(" ".join(post_text)).strip()]
    return "\n\n".join(p for p in parts if p)


def _cell_number(text: str) -> Optional[float]:
    """First number in a cell: "$ 5735" -> 5735, "22% ( 22 % )" -> 22,
    "( 41317 )" -> 41317 (parentheses are magnitudes in FinQA's programs)."""
    m = _PLAIN_NUMBER.search(str(text).replace(",", "").replace("$", ""))
    return float(m.group()) if m else None


def grounding_cells(table: list[list[str]], program_dsl: str) -> list[dict]:
    """Table cells the gold program reads: cells whose number is a literal operand,
    plus every numeric cell of a row used by a table_* op. For the faithfulness
    grounding check (proposal 5.5)."""
    program_dsl = program_dsl or ""
    if "(" not in program_dsl:  # a bare-number lookup turn
        v = _cell_number(program_dsl)
        return _match_cells(table, {v} if v is not None else set(), set())
    toks = _tokenize(program_dsl)
    operands: set[float] = set()
    rows_used: set[str] = set()
    for i in range(0, len(toks) - 3, 4):
        op = toks[i][:-1].strip()
        if op in _TABLE_OPS:
            rows_used.add(toks[i + 1].strip())
            continue
        for a in (toks[i + 1], toks[i + 2]):
            a = a.strip()
            if a.startswith("#") or a.startswith("const_"):
                continue
            v = _cell_number(a)
            if v is not None:
                operands.add(v)
    return _match_cells(table, operands, rows_used)


def _match_cells(table: list[list[str]], operands: set[float],
                 rows_used: set[str]) -> list[dict]:
    cells = []
    for r, row in enumerate(table):
        in_used_row = bool(row) and str(row[0]).strip() in rows_used
        for c, cell in enumerate(row):
            v = _cell_number(cell)
            if v is None or c == 0:
                continue
            if in_used_row or v in operands:
                cells.append({"row": r, "col": c, "text": str(cell)})
    return cells


# --- per-dataset builders: raw dir -> {split: [record, ...]} ---

def _finqa_like_record(e: dict, uid: str, question: str, program_dsl: str,
                       gold: object, meta: dict) -> dict:
    table = e["table"]
    try:
        code = finqa_program_to_python(program_dsl, table)
        ok = verified_program(code, gold)
    except ProgramError:
        code, ok = None, False
    meta = {**meta, "program_dsl": program_dsl, "program_verified": ok,
            "filename": e.get("filename"), "table": normalize_table(table)}
    return {"uid": uid, "question": question,
            "context": build_context(e.get("pre_text", []), table, e.get("post_text", [])),
            "answer": gold, "gold_program": code if ok else None,
            "table_cells": grounding_cells(table, program_dsl), "meta": meta}


def build_finqa(raw_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for split in ("train", "dev", "test"):
        raw = json.loads((raw_dir / "finqa" / f"{split}.json").read_text(encoding="utf-8"))
        out[split] = [
            _finqa_like_record(
                e, uid=e["id"], question=e["qa"]["question"],
                program_dsl=e["qa"]["program"], gold=e["qa"]["exe_ans"],
                meta={"dataset": "finqa", "source_split": split,
                      "answer_text": e["qa"].get("answer", ""),
                      "gold_inds": sorted(e["qa"].get("gold_inds", {}))})
            for e in raw
        ]
    return out


def build_convfinqa(raw_dir: Path) -> dict[str, list[dict]]:
    """Turn-level records. `question` is the current turn; earlier turns (with
    their gold answers) are in meta.history for the prompt builder to render."""
    z = zipfile.ZipFile(raw_dir / "convfinqa" / "data.zip")
    out = {}
    for split in ("train", "dev"):  # test answers are private
        raw = json.loads(z.read(f"data/{split}_turn.json"))
        recs = []
        for e in raw:
            a = e["annotation"]
            turn = len(a["cur_dial"]) - 1
            history = [{"question": q, "answer": ans}
                       for q, ans in zip(a["cur_dial"][:turn], a["exe_ans_list"][:turn])]
            recs.append(_finqa_like_record(
                e, uid=e["id"], question=a["cur_dial"][turn],
                program_dsl=a["cur_program"], gold=a["exe_ans"],
                meta={"dataset": "convfinqa", "source_split": split,
                      "conversation_id": e["id"].rsplit("_", 1)[0], "turn_index": turn,
                      "history": history, "turn_type": a.get("cur_type"),
                      "qa_split": a.get("qa_split")}))
        out[split] = recs
    return out


def _tatqa_answer(q: dict) -> object:
    ans = q["answer"]
    if q["answer_type"] in ("arithmetic", "count"):
        try:
            v = float(ans)
            return int(v) if v.is_integer() else v
        except (TypeError, ValueError):
            return ans
    if isinstance(ans, list):
        return ans[0] if len(ans) == 1 else "; ".join(str(x) for x in ans)
    return ans


def _tatqa_program(q: dict, gold: object) -> Optional[str]:
    """Verified Python for an arithmetic item. TAT-QA percent answers are in
    percent units (13.67 for 13.67%) while derivations usually compute the ratio,
    so for scale == "percent" a x100 variant is also tried."""
    if q["answer_type"] != "arithmetic":
        return None
    try:
        expr = tatqa_derivation_to_expr(q.get("derivation", ""))
    except ProgramError:
        return None
    candidates = [f"answer = {expr}"]
    if q.get("scale") == "percent":
        candidates.append(f"answer = ({expr}) * 100")
    return next((c for c in candidates if verified_program(c, gold, round_2dp=True)), None)


def _tatqa_grounding(table: list[list[str]], code: Optional[str]) -> list[dict]:
    if not code:
        return []
    operands = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", code.split("=", 1)[1])}
    operands.discard(100.0)  # the percent conversion constant
    return _match_cells(table, operands, set())


def build_tatqa(raw_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for split, fname in (("train", "train.json"), ("dev", "dev.json"),
                         ("test", "test_gold.json")):
        raw = json.loads((raw_dir / "tatqa" / fname).read_text(encoding="utf-8"))
        recs = []
        for doc in raw:
            table = doc["table"]["table"]
            paras = [normalize_numbers(p["text"]).strip()
                     for p in sorted(doc["paragraphs"], key=lambda p: p["order"])]
            context = "\n\n".join(x for x in [linearize_table(normalize_table(table)),
                                              "\n".join(paras)] if x)
            for q in sorted(doc["questions"], key=lambda q: q["order"]):
                gold = _tatqa_answer(q)
                code = _tatqa_program(q, gold)
                recs.append({
                    "uid": q["uid"], "question": q["question"], "context": context,
                    "answer": gold, "gold_program": code,
                    "table_cells": _tatqa_grounding(table, code),
                    "meta": {"dataset": "tatqa", "source_split": split,
                             "answer_type": q["answer_type"],
                             "answer_from": q.get("answer_from"),
                             "scale": q.get("scale", ""), "answer_raw": q["answer"],
                             "derivation": q.get("derivation", ""),
                             "program_verified": code is not None,
                             "req_comparison": q.get("req_comparison"),
                             "table_uid": doc["table"]["uid"],
                             "table": normalize_table(table)},
                })
        out[split] = recs
    return out


BUILDERS: dict[str, Callable[[Path], dict[str, list[dict]]]] = {
    "finqa": build_finqa, "convfinqa": build_convfinqa, "tatqa": build_tatqa,
}
