"""Run reports: numbered figures and a one-page summary of a runs directory.

`build_report(runs_dir)` writes runs_dir/report/: PNG figures, index.html (open
it in a browser) and report.md. It reads only what the runner logs
(results.jsonl; per run predictions.jsonl, train_stats.json, status.json) and
joins predictions with the dataset by uid, when the data is available, for the
question-type breakdowns.

The latest completed run gets a deep dive (figures 01-13); figures 14-18
compare runs and appear once there is more than one to compare.

The answer-matching variants (`match_variants`) are DIAGNOSTIC: the official
score stays `ror.metrics.answers_match`. They show how much a score depends on
the matching rule (percent form, unevaluated arithmetic, rounding), which the
study has to settle before arms are compared.
"""
from __future__ import annotations

import ast
import html
import json
import math
import operator as op
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from .metrics import answers_match, normalize_number

# Palette: the dataviz reference instance (light surface; figures are static).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"

REPORT_DIR = "report"

# Cumulative matching rules, strictest first (diagnostic only).
RULES = [
    ("strict", "Current rule (strict)"),
    ("percent", "+ '14.5%' read as 0.145"),
    ("arithmetic", "+ arithmetic like '301/2575' evaluated"),
    ("within_1pct", "+ within 1% of the gold value"),
    ("human_rounding", "+ equal to FinQA's rounded human answer"),
]

OUTCOMES = [
    "Correct",
    "Right value, written as a %",
    "Right arithmetic, not evaluated",
    "Close (within 1%)",
    "Sign flipped",
    "Off by a power of 10",
    "Wrong yes/no",
    "Wrong number",
    "Not a number",
    "No answer found",
]


# --- answer analysis (pure Python) ----------------------------------------------

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
_CMP = {ast.Gt: op.gt, ast.Lt: op.lt, ast.GtE: op.ge, ast.LtE: op.le}
_HAS_OPERATOR = re.compile(r"[\d)]\s*[-+*/<>]=?\s*[-\d(.]")


def eval_arithmetic(text: Optional[str]) -> Optional[object]:
    """The value of a plain arithmetic answer such as '301/2575' (a float) or
    '286.61 - 198.09 > 0' ('yes'/'no'): numbers, + - * /, parentheses and one
    comparison. Anything else (names, calls, plain numbers) gives None."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    s = (first.replace("$", "").replace(",", "").replace("%", "")
         .replace("−", "-").replace("×", "*").replace("÷", "/").strip())
    if not _HAS_OPERATOR.search(s):
        return None
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("not plain arithmetic")

    try:
        body = tree.body
        if isinstance(body, ast.Compare) and len(body.ops) == 1 and type(body.ops[0]) in _CMP:
            return "yes" if _CMP[type(body.ops[0])](ev(body.left), ev(body.comparators[0])) \
                else "no"
        return ev(body)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _first_line(text: Optional[str]) -> str:
    t = (text or "").strip()
    return t.splitlines()[0] if t else ""


def _candidates(row: dict) -> tuple[list, list, list]:
    """(as parsed, percent-corrected, arithmetic-evaluated) readings of an answer."""
    pred, first = row.get("pred"), _first_line(row.get("text"))
    pct = "%" in first
    pct_c = [pred / 100] if pct and isinstance(pred, (int, float)) else []
    ex = eval_arithmetic(first)
    expr_c: list = []
    if ex is not None:
        expr_c.append(ex)
        if pct and isinstance(ex, float):
            expr_c.append(ex / 100)
    return [pred], pct_c, expr_c


def _within(value: object, gold: object, rel: float) -> bool:
    v, g = normalize_number(value), normalize_number(gold)
    if v is None or g is None or isinstance(value, str) and isinstance(gold, str):
        return False
    return abs(v - g) <= max(1e-4, rel * abs(g))


def _human_match(value: object, answer_text: Optional[str]) -> bool:
    """True if `value` rounds to the annotator's written answer ('14%', '9.9%',
    '94'), at the precision the annotator wrote it."""
    v = normalize_number(value) if not isinstance(value, str) else None
    m = re.search(r"-?\d+(?:\.\d+)?", (answer_text or "").replace(",", ""))
    if v is None or not m:
        return False
    num = m.group()
    decimals = len(num.split(".")[1]) if "." in num else 0
    target, half = float(num), 0.5 * 10 ** -decimals
    if "%" in answer_text:
        target, half = target / 100, half / 100
    return abs(v - target) <= half + 1e-12


def match_variants(row: dict, answer_text: Optional[str] = None) -> dict[str, bool]:
    """Whether an answer counts as correct under each rule in RULES (cumulative:
    each rule accepts everything the previous one does)."""
    gold = row.get("gold")
    base, pct_c, expr_c = _candidates(row)
    out: dict[str, bool] = {}
    out["strict"] = any(answers_match(c, gold) for c in base)
    out["percent"] = out["strict"] or any(answers_match(c, gold) for c in pct_c)
    readings = base + pct_c + expr_c
    out["arithmetic"] = out["percent"] or any(answers_match(c, gold) for c in expr_c)
    out["within_1pct"] = out["arithmetic"] or any(_within(c, gold, 0.01) for c in readings)
    out["human_rounding"] = out["within_1pct"] or (
        bool(answer_text) and any(_human_match(c, answer_text) for c in readings))
    return out


def outcome(row: dict) -> str:
    """One label from OUTCOMES describing what happened to an answer."""
    gold, pred = row.get("gold"), row.get("pred")
    if row.get("correct"):
        return "Correct"
    if pred is None:
        return "No answer found"
    base, pct_c, expr_c = _candidates(row)
    if any(answers_match(c, gold) for c in pct_c):
        return "Right value, written as a %"
    if any(answers_match(c, gold) for c in expr_c):
        return "Right arithmetic, not evaluated"
    if isinstance(gold, str) and gold.lower() in ("yes", "no"):
        return "Wrong yes/no"
    numbers = [c for c in base + pct_c + expr_c if isinstance(c, (int, float))]
    g = normalize_number(gold)
    if g is None or not numbers:
        return "Not a number"
    if any(_within(c, g, 0.01) for c in numbers):
        return "Close (within 1%)"
    if any(_within(-c, g, 0.01) for c in numbers):
        return "Sign flipped"
    if g != 0 and any(c != 0 and _power_of_ten_off(c, g) for c in numbers):
        return "Off by a power of 10"
    return "Wrong number"


def _power_of_ten_off(value: float, gold: float) -> bool:
    ratio = abs(value / gold)
    k = round(math.log10(ratio))
    return k != 0 and abs(ratio / 10 ** k - 1) <= 0.01


_OP = re.compile(r"([a-z_]+)\(")


def question_features(example: Any) -> dict:
    """What kind of question an Example is: reasoning steps and operations (from
    the gold program), where the evidence sits, and the human-written answer."""
    meta = getattr(example, "meta", None) or {}
    ops = _OP.findall(meta.get("program_dsl") or "")
    inds = meta.get("gold_inds") or []
    keys = list(inds.keys()) if isinstance(inds, dict) else [str(k) for k in inds]
    table = any(k.startswith("table") for k in keys)
    text = any(k.startswith("text") for k in keys)
    if not keys and meta.get("answer_from"):          # TAT-QA
        src = str(meta["answer_from"])
        table, text = "table" in src, "text" in src
    evidence = ("table + text" if table and text else "table only" if table
                else "text only" if text else "unknown")
    return {"question": getattr(example, "question", ""),
            "n_steps": len(ops) or None, "ops": sorted(set(ops)), "evidence": evidence,
            "answer_text": meta.get("answer_text")}


# --- loading ------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_runs(runs_dir: Path) -> list[dict]:
    """Result rows (latest per experiment), each with its run folder's files."""
    latest: dict[str, dict] = {}
    for r in _read_jsonl(Path(runs_dir) / "results.jsonl"):
        if r.get("exp_id") and (r["exp_id"] not in latest or
                                r.get("timestamp", 0) >= latest[r["exp_id"]].get("timestamp", 0)):
            latest[r["exp_id"]] = r
    runs = sorted(latest.values(), key=lambda r: r.get("timestamp", 0))
    for r in runs:
        d = Path(runs_dir) / r["exp_id"]
        r["_predictions"] = _read_jsonl(d / "predictions.jsonl")
        r["_train"] = (r.get("extra") or {}).get("train") or _read_json(d / "train_stats.json") or {}
    return runs


def _examples_by_uid(dataset: str, split: str) -> dict:
    try:
        from .data import load_dataset

        return {ex.uid: ex for ex in load_dataset(dataset, split)}
    except Exception:  # noqa: BLE001 — data not available here: skip breakdowns
        return {}


def _statuses(runs_dir: Path) -> Counter:
    counts: Counter = Counter()
    for f in Path(runs_dir).glob("*/status.json"):
        st = _read_json(f)
        if st and st.get("status"):
            counts[st["status"]] += 1
    return counts


# --- figure helpers -----------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 10,
        "legend.frameon": False, "text.color": INK, "axes.titlesize": 10.5,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def _figure(plt, title: str, subtitle: str, height: float = 4.2, width: float = 8.6,
            ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    top = 1 - 0.95 / height
    fig.subplots_adjust(top=top, left=0.1, right=0.97, bottom=0.14, wspace=0.45)
    fig._ror_titles = [  # type: ignore[attr-defined]
        fig.text(0.012, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13,
                 fontweight="bold", color=INK),
        fig.text(0.012, 1 - 0.47 / height, subtitle, ha="left", va="top", fontsize=9.3,
                 color=INK2)]
    return fig, axes


def _save(plt, fig, path: Path) -> Path:
    titles = getattr(fig, "_ror_titles", [])
    if titles and fig.axes:
        # align the title block with the left edge of the (tick-labelled) plot
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        x0 = min(inv.transform(ax.get_tightbbox(renderer))[0][0]
                 for ax in fig.axes if ax.get_visible() and ax.axison)
        for t in titles:
            t.set_x(x0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def _percent_axis(ax) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _hbars(ax, labels: list[str], values: list[float], text: list[str],
           color: str = SERIES[0], xmax: Optional[float] = None,
           percent: bool = False) -> None:
    """Horizontal bars with a value label at each bar end. `percent` puts the
    values on a fixed 0-100% scale so charts of different runs compare."""
    if percent:
        xmax = 1.0
    ys = list(range(len(labels)))
    ax.barh(ys, values, height=0.62, color=color)
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    top = xmax if xmax is not None else max(values + [1e-9])
    ax.set_xlim(0, top * 1.22 if top > 0 else 1)
    for y, v, t in zip(ys, values, text):
        ax.text(v + top * 0.015, y, t, va="center", ha="left", fontsize=9, color=INK2)
    if percent:
        _percent_axis(ax)


def _legend_below(ax, ncol: int) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=ncol, handlelength=1.2)


def _integer_axis(axis) -> None:
    from matplotlib.ticker import MaxNLocator

    axis.set_major_locator(MaxNLocator(integer=True))


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _si(x: Optional[float], unit: str = "FLOP") -> str:
    if not x:
        return "n/a"
    for p, name in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M")):
        if abs(x) >= p:
            return f"{x / p:.1f} {name}{unit}"
    return f"{x:.0f} {unit}"


def _minutes(s: Optional[float]) -> str:
    if s is None:
        return "n/a"
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def _accuracy_by(items: list[dict], key: Callable[[dict], Optional[str]],
                 order: Optional[list[str]] = None) -> tuple[list[str], list[float], list[str]]:
    groups: dict[str, list[bool]] = {}
    for it in items:
        k = key(it)
        if k is not None:
            groups.setdefault(k, []).append(bool(it["correct"]))
    labels = [k for k in (order or sorted(groups)) if k in groups]
    acc = [sum(groups[k]) / len(groups[k]) for k in labels]
    text = [f"{100 * a:.0f}%  (n={len(groups[k])})" for a, k in zip(acc, labels)]
    return labels, acc, text


# --- figures: the latest run --------------------------------------------------

def fig_scorecard(plt, run: dict, items: list[dict], arm_note: str, path: Path) -> Path:
    tr, timing = run["_train"], (run.get("extra") or {}).get("timing_s") or {}
    n = run.get("n_examples") or len(items)
    n_ok = sum(bool(i["correct"]) for i in items)
    strict = run.get("exact_match_strict")
    tiles = [("Exact match (primary rule)", _pct(run.get("exact_match")),
              f"{n_ok} of {n} test questions" + (f"; strict {_pct(strict)}"
                                                 if strict is not None else ""))]
    if tr:
        ev = [d["loss"] for d in tr.get("dev_loss") or []] or tr.get("eval_loss_by_epoch") or []
        tiles.append(("Loss", f"{tr.get('train_loss', float('nan')):.2f}"
                      + (f" -> {ev[-1]:.2f}" if ev else ""),
                      "training average -> dev set" if ev else "training average"))
        tiles.append(("Training speed", f"{tr.get('tokens_per_s') or 0:,.0f} tok/s",
                      f"{tr.get('n_train')} questions, {tr.get('global_steps')} steps, "
                      f"{tr.get('epochs')} epoch(s)"))
    infer_s = timing.get("inference")
    if infer_s:
        tiles.append(("Answering speed", f"{n / infer_s:.2f} q/s",
                      f"{n} questions in {_minutes(infer_s)}"))
    else:
        tiles.append(("Compute", _si((run.get("train_flops") or 0)
                                     + (run.get("infer_flops") or 0)),
                      "training + answering (estimated)"))
    tiles.append(("Wall time", _minutes(run.get("wall_time_s")), "whole experiment"))
    mem = tr.get("peak_gpu_mem_gb") if tr else None
    tiles.append(("Hardware", str(run.get("gpu") or "n/a"),
                  f"peak {mem} GB GPU memory" if mem else
                  f"{tr.get('compute_dtype', '')} compute, 4-bit weights" if tr else ""))
    fig = plt.figure(figsize=(8.6, 3.6))
    fig.text(0.012, 0.97, run["name"], ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.87, arm_note, ha="left", va="top", fontsize=9.3, color=INK2)
    cols, rows = 3, math.ceil(len(tiles) / 3)
    for i, (label, value, sub) in enumerate(tiles):
        x = 0.012 + (i % cols) * 0.335
        y = 0.70 - (i // cols) * (0.62 / max(rows, 1)) - 0.02
        fig.text(x, y, label, fontsize=9, color=INK2, va="top")
        fig.text(x, y - 0.07, value, fontsize=19, color=INK, va="top")
        fig.text(x, y - 0.21, sub, fontsize=8.3, color=MUTED, va="top")
    return _save(plt, fig, path)


def fig_training_curve(plt, run: dict, path: Path) -> Optional[Path]:
    tr = run["_train"]
    hist = tr.get("log_history") or []
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h and "step" in h]
    evals = ([(d["step"], d["loss"]) for d in tr.get("dev_loss") or []]
             or [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h and "step" in h])
    selection = tr.get("selection") or {}
    checks = selection.get("checks") or []
    if len(train) < 2 and not checks:
        return None
    if checks:
        fig, (ax, a2) = _figure(plt, "Training and checkpoint selection",
                                "Left: loss on the answer tokens per step; dots = dev-set "
                                "loss. Right: dev exact match of each kept checkpoint; the "
                                "ringed one is used.", ncols=2)
    else:
        fig, ax = _figure(plt, "Training loss",
                          "Loss on the answer tokens per optimizer step; dots = dev-set "
                          "loss. Lower is better.")
    if len(train) >= 2:
        ax.plot(*zip(*train), color=SERIES[0], linewidth=2, label="training")
    if evals:
        ax.plot(*zip(*evals), linestyle="none", marker="o", markersize=7, color=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="dev set")
        ax.legend(loc="upper right")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    if checks:
        steps = [c["step"] for c in checks]
        ems = [c["dev_em"] for c in checks]
        a2.plot(steps, ems, color=SERIES[0], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        chosen = selection.get("chosen_step")
        if chosen in steps:
            y = ems[steps.index(chosen)]
            a2.plot([chosen], [y], marker="o", markersize=15, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.5)
            a2.annotate(f"kept: {_pct(y)}", (chosen, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8.5, color=INK2)
        a2.set_ylim(0, 1.0)
        from matplotlib.ticker import PercentFormatter

        a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        a2.set_xlabel("optimizer step")
        a2.set_title("dev exact match")
    return _save(plt, fig, path)


def fig_lr_grad(plt, run: dict, path: Path) -> Optional[Path]:
    hist = run["_train"].get("log_history") or []
    lr = [(h["step"], h["learning_rate"]) for h in hist if "learning_rate" in h]
    gn = [(h["step"], h["grad_norm"]) for h in hist if "grad_norm" in h]
    if len(lr) < 2 and len(gn) < 2:
        return None
    fig, (a1, a2) = _figure(plt, "Learning rate and gradient size",
                            "Left: the warm-up then cosine schedule actually used. Right: "
                            "gradient norm per step (spikes = unstable training).",
                            ncols=2)
    for ax, pts, name in ((a1, lr, "learning rate"), (a2, gn, "gradient norm")):
        if len(pts) >= 2:
            ax.plot(*zip(*pts), color=SERIES[0], linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("optimizer step")
    a1.yaxis.set_major_formatter(lambda x, _: f"{x:.1e}")
    return _save(plt, fig, path)


def fig_pred_vs_gold(plt, items: list[dict], path: Path) -> Optional[Path]:
    pts = [(normalize_number(i["gold"]), normalize_number(i["pred"]), bool(i["correct"]))
           for i in items if not isinstance(i["gold"], str)]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        return None
    fig, ax = _figure(plt, "Model answer vs correct answer",
                      "Each dot is a numeric test question. On the diagonal = right value. "
                      "Parallel off-diagonal bands = answers x100 (percent form) or /100.",
                      height=5.2, width=7.2)
    lim = max(max(abs(g), abs(p)) for g, p, _ in pts) * 1.5
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, linewidth=1, zorder=1)
    for ok, color, marker, label in ((True, GOOD, "o", "correct"),
                                     (False, CRITICAL, "X", "wrong")):
        sel = [(g, p) for g, p, c in pts if c == ok]
        if sel:
            ax.scatter(*zip(*sel), s=46, color=color, marker=marker, edgecolors=SURFACE,
                       linewidths=1.2, label=f"{label} ({len(sel)})", zorder=3)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("correct (gold) answer")
    ax.set_ylabel("model answer")
    ax.legend(loc="upper left")
    return _save(plt, fig, path)


def fig_outcomes(plt, items: list[dict], path: Path) -> Path:
    counts = Counter(i["_outcome"] for i in items)
    labels = [o for o in OUTCOMES if counts[o]]
    vals = [counts[o] for o in labels]
    n = len(items)
    fig, ax = _figure(plt, "What happened to each answer",
                      "Every test question falls in exactly one group. Only 'Correct' "
                      "scores; the next groups are right in substance but not in form.",
                      height=0.9 + 0.42 * max(len(labels), 3))
    _hbars(ax, labels, vals, [f"{v}  ({100 * v / n:.0f}%)" for v in vals])
    ax.set_xlabel("questions")
    _integer_axis(ax.xaxis)
    return _save(plt, fig, path)


def fig_rules(plt, items: list[dict], path: Path) -> Path:
    labels = [name for _, name in RULES]
    vals = [sum(i["_variants"][key] for i in items) / len(items) for key, _ in RULES]
    fig, ax = _figure(plt, "Score under different answer-matching rules (diagnostic)",
                      "Each bar also accepts everything above it. The study reports the "
                      "first bar until the rule is decided; the gap is what is at stake.",
                      height=3.4)
    _hbars(ax, labels, vals, [_pct(v) for v in vals], percent=True)
    return _save(plt, fig, path)


def fig_error_size(plt, items: list[dict], path: Path) -> Optional[Path]:
    errs = []
    for i in items:
        g, p = normalize_number(i["gold"]), normalize_number(i["pred"])
        if g not in (None, 0) and p is not None and not isinstance(i["gold"], str):
            errs.append(max(-4.0, min(3.0, math.log10(max(abs(p - g) / abs(g), 1e-4)))))
    if len(errs) < 2:
        return None
    fig, ax = _figure(plt, "How far off the numeric answers are",
                      "Relative error |model - gold| / |gold|, log scale. Left of the 1% "
                      "line = near misses; around 100% and beyond = a different quantity.")
    bins = [x / 4 for x in range(-16, 13)]
    ax.hist(errs, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    _integer_axis(ax.yaxis)
    for x, label in ((-2, "1%"), (-1, "10%"), (0, "100%")):
        ax.axvline(x, color=MUTED, linewidth=1)
        ax.text(x + 0.05, ax.get_ylim()[1] * 0.95, label, color=INK2, fontsize=8.5, va="top")
    ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3],
                  ["<=0.01%", "0.1%", "1%", "10%", "100%", "10x", "100x", ">=1000x"])
    ax.set_xlabel("relative error")
    ax.set_ylabel("questions")
    ax.grid(axis="x", visible=False)
    return _save(plt, fig, path)


def _breakdown(plt, items: list[dict], key: Callable[[dict], Optional[str]], title: str,
               subtitle: str, path: Path, order: Optional[list[str]] = None) -> Optional[Path]:
    labels, acc, text = _accuracy_by(items, key, order)
    if len(labels) < 2:
        return None
    fig, ax = _figure(plt, title, subtitle, height=1.0 + 0.45 * max(len(labels), 3))
    _hbars(ax, labels, acc, text, percent=True)
    ax.set_xlabel("exact match")
    return _save(plt, fig, path)


def _length_buckets(items: list[dict]) -> tuple[Callable[[dict], Optional[str]], list[str]]:
    """Quartile buckets of prompt length: (key function, labels in order)."""
    lens = sorted(i.get("prompt_tokens") or 0 for i in items)
    if not lens or lens[-1] == 0:
        return (lambda i: None), []
    edges = sorted({lens[min(len(lens) - 1, int(len(lens) * f))] for f in (0.25, 0.5, 0.75)})
    labels, lo = [], 0
    for e in edges:
        labels.append(f"{lo:,}-{e:,} tokens")
        lo = e + 1
    labels.append(f"> {edges[-1]:,} tokens")

    def key(i: dict) -> Optional[str]:
        n = i.get("prompt_tokens") or 0
        for e, label in zip(edges, labels):
            if n <= e:
                return label
        return labels[-1]
    return key, labels


def fig_answer_length(plt, items: list[dict], path: Path) -> Optional[Path]:
    lens = [i.get("gen_tokens") for i in items if i.get("gen_tokens") is not None]
    if len(lens) < 2:
        return None
    fig, ax = _figure(plt, "Length of the model's replies",
                      "Generated tokens per question. Answer-only models should reply in a "
                      "few tokens; long replies mean the format was not learned.")
    top = max(lens)
    bins = [k - 0.5 for k in range(0, top + 2)] if top <= 40 else 30
    ax.hist(lens, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    _integer_axis(ax.xaxis)
    _integer_axis(ax.yaxis)
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("questions")
    ax.grid(axis="x", visible=False)
    return _save(plt, fig, path)


def fig_time_compute(plt, run: dict, path: Path) -> Optional[Path]:
    timing = (run.get("extra") or {}).get("timing_s") or {}
    tr = run["_train"]
    wall = run.get("wall_time_s") or 0
    if not wall:
        return None
    parts: list[tuple[str, float]] = []
    if timing:
        train_s = tr.get("train_runtime_s_this_session") or 0
        load_s = max(0.0, (timing.get("model_and_training") or 0) - train_s)
        parts = [("loading model", load_s), ("training", train_s),
                 ("answering", timing.get("inference") or 0)]
        parts.append(("other", max(0.0, wall - sum(v for _, v in parts))))
    else:
        train_s = tr.get("train_runtime_s_this_session") or 0
        parts = [("training", train_s), ("everything else", max(0.0, wall - train_s))]
    parts = [(k, v) for k, v in parts if v > 0]
    flops = [(k, v) for k, v in (("training", run.get("train_flops")),
                                 ("answering", run.get("infer_flops"))) if v]
    fig, (a1, a2) = _figure(plt, "Where the time and compute went",
                            "Left: wall-clock minutes of this experiment. Right: estimated "
                            "floating-point operations (the x-axis of the compute frontier).",
                            ncols=2, height=3.4)
    _hbars(a1, [k for k, _ in parts], [v / 60 for _, v in parts],
           [_minutes(v) for _, v in parts])
    a1.set_xlabel("minutes")
    if flops:
        _hbars(a2, [k for k, _ in flops], [v for _, v in flops], [_si(v) for _, v in flops])
        a2.set_xlabel("floating-point operations")
        from matplotlib.ticker import MaxNLocator

        a2.xaxis.set_major_locator(MaxNLocator(3))
        a2.xaxis.set_major_formatter(lambda x, _: _si(x).replace(".0 ", " ") if x else "0")
    else:
        a2.set_axis_off()
    return _save(plt, fig, path)


# --- figures: across runs -----------------------------------------------------

def _run_label(r: dict) -> str:
    return f"{r['arm']} {r['model']}" + (f" s{r['seed']}" if r.get("seed") else "")


def fig_all_runs(plt, runs: list[dict], path: Path) -> Optional[Path]:
    if len(runs) < 2:
        return None
    labels = sorted({_run_label(r) for r in runs})
    fig, ax = _figure(plt, "Exact match of every completed experiment",
                      "Standard test split vs the contamination-controlled (clean) set. "
                      "The gap between the two bars is the contamination gap.",
                      height=1.1 + 0.5 * max(len(labels), 3))
    splits = [s for s in ("standard", "clean") if any(r["split"] == s for r in runs)]
    h = 0.8 / len(splits)
    for j, s in enumerate(splits):
        by = {_run_label(r): r.get("exact_match") for r in runs if r["split"] == s}
        ys = [i + (j - (len(splits) - 1) / 2) * h for i in range(len(labels))]
        vals = [by.get(lb) or 0 for lb in labels]
        ax.barh(ys, vals, height=h * 0.9, color=SERIES[j], label=s)
        for y, lb, v in zip(ys, labels, vals):
            if lb in by:
                ax.text(v + 0.01, y, _pct(by[lb]), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.22)
    _percent_axis(ax)
    _legend_below(ax, len(splits))
    return _save(plt, fig, path)


def fig_frontier(plt, runs: list[dict], path: Path) -> Optional[Path]:
    pts = [r for r in runs if r.get("infer_flops") and r.get("exact_match") is not None
           and r.get("n_examples")]
    if len(pts) < 2:
        return None
    fig, ax = _figure(plt, "Accuracy vs inference compute",
                      "Each dot is one experiment: inference FLOPs per question vs exact "
                      "match. Up and to the left is better (more accuracy for less compute).",
                      height=4.8)
    for j, s in enumerate(("standard", "clean")):
        sel = [r for r in pts if r["split"] == s]
        if sel:
            ax.scatter([r["infer_flops"] / r["n_examples"] for r in sel],
                       [r["exact_match"] for r in sel], s=60, color=SERIES[j],
                       edgecolors=SURFACE, linewidths=1.5, label=s, zorder=3)
            if len(pts) <= 12:
                for r in sel:
                    ax.annotate(_run_label(r), (r["infer_flops"] / r["n_examples"],
                                                r["exact_match"]),
                                textcoords="offset points", xytext=(6, 4), fontsize=8,
                                color=INK2)
    from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(lambda x, _: _si(x).replace(".0 ", " "))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("inference FLOPs per question")
    ax.set_ylabel("exact match")
    _legend_below(ax, 2)
    return _save(plt, fig, path)


def fig_gap(plt, runs: list[dict], path: Path) -> Optional[Path]:
    std = {(r["arm"], r["model"], r.get("seed")): r for r in runs if r["split"] == "standard"}
    gaps = []
    for r in runs:
        if r["split"] == "clean":
            s = std.get((r["arm"], r["model"], r.get("seed")))
            if s and s.get("exact_match") is not None and r.get("exact_match") is not None:
                gaps.append((_run_label(r), s["exact_match"] - r["exact_match"]))
    if len(gaps) < 2:          # one gap is a number, not a chart (it is in the tables)
        return None
    fig, ax = _figure(plt, "Contamination gap",
                      "Exact match on the standard split minus on the clean set, in "
                      "percentage points. Bigger = more of the standard score is recall.",
                      height=1.1 + 0.45 * max(len(gaps), 3))
    labels, vals = zip(*gaps)
    ax.barh(range(len(vals)), [100 * v for v in vals], height=0.62, color=SERIES[0])
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo, hi = min(0.0, 100 * min(vals)), max(0.0, 100 * max(vals))
    ax.set_xlim(lo * 1.25 - 1, hi * 1.25 + 1)
    for y, v in enumerate(vals):
        ax.text(100 * v, y, f" {100 * v:+.1f} pts", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.set_xlabel("percentage points")
    return _save(plt, fig, path)


def fig_faithfulness(plt, runs: list[dict], path: Path) -> Optional[Path]:
    prog = [r for r in runs if r.get("executability_rate") is not None]
    if not prog:
        return None
    metrics = [("execution_accuracy", "execution accuracy"),
               ("executability_rate", "program runs"),
               ("faithfulness_primary", "faithfulness")]
    labels = [f"{_run_label(r)} ({r['split']})" for r in prog]
    fig, ax = _figure(plt, "Program arms: do right answers come from working programs?",
                      "Execution accuracy = program's result is right; program runs = it "
                      "executes at all; faithfulness = share of right answers a program produced.",
                      height=1.2 + 0.6 * max(len(prog), 2))
    h = 0.8 / len(metrics)
    for j, (key, name) in enumerate(metrics):
        ys = [i + (j - 1) * h for i in range(len(prog))]
        vals = [r.get(key) or 0 for r in prog]
        ax.barh(ys, vals, height=h * 0.9, color=SERIES[j], label=name)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, _pct(v), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(prog)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.18)
    _percent_axis(ax)
    _legend_below(ax, len(metrics))
    return _save(plt, fig, path)


def fig_status(plt, counts: Counter, path: Path) -> Optional[Path]:
    if sum(counts.values()) < 2:
        return None
    color = {"completed": GOOD, "failed": CRITICAL, "permanently_failed": CRITICAL,
             "pending": WARNING, "running": SERIES[0]}
    labels = [s for s in ("completed", "running", "pending", "failed", "permanently_failed")
              if counts[s]]
    fig, ax = _figure(plt, "Experiment registry",
                      "Status of every experiment in this runs folder. Completed and "
                      "permanently failed ones are never run again.", height=3.2)
    ys = list(range(len(labels)))
    ax.barh(ys, [counts[s] for s in labels], height=0.62, color=[color[s] for s in labels])
    ax.set_yticks(ys, [s.replace("_", " ") for s in labels])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for y, s in zip(ys, labels):
        ax.text(counts[s], y, f"  {counts[s]}", va="center", fontsize=9, color=INK2)
    ax.set_xlabel("experiments")
    return _save(plt, fig, path)


# --- the report ---------------------------------------------------------------

CAPTIONS = {
    "01": "Headline numbers of the latest completed experiment.",
    "02": "Training loss per step with dev loss, and the dev accuracy of each kept checkpoint.",
    "03": "Learning-rate schedule and gradient norm.",
    "04": "Model answer against the correct answer for every numeric question.",
    "05": "What happened to each answer.",
    "06": "The same answers scored under progressively more lenient matching rules.",
    "07": "Size of the numeric errors.",
    "08": "Accuracy by the number of calculation steps in the gold program.",
    "09": "Accuracy by the operations the gold program uses.",
    "10": "Accuracy by where the needed numbers are (table, text, or both).",
    "11": "Accuracy by prompt length.",
    "12": "Length of the model's replies.",
    "13": "Where the time and compute went.",
    "14": "Every completed experiment, standard vs clean.",
    "15": "Accuracy vs inference compute (the frontier).",
    "16": "Contamination gap per experiment.",
    "17": "Program arms: execution accuracy, executability, faithfulness.",
    "18": "Registry status of all experiments.",
}


def build_report(runs_dir: str | Path, out_dir: Optional[str | Path] = None) -> list[Path]:
    """Write the figures, index.html and report.md for `runs_dir`; return the files.
    Without matplotlib the pages are still written (tables only) and say so."""
    runs_dir = Path(runs_dir)
    out = Path(out_dir) if out_dir else runs_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    runs = load_runs(runs_dir)
    statuses = _statuses(runs_dir)

    focus = next((r for r in reversed(runs) if r["_predictions"]), None)
    items: list[dict] = []
    if focus:
        by_uid = _examples_by_uid(focus.get("dataset", "finqa"), focus.get("split", "standard"))
        for row in focus["_predictions"]:
            feats = question_features(by_uid[row["uid"]]) if row.get("uid") in by_uid else {}
            items.append({**row, **{f"_{k}": v for k, v in feats.items()},
                          "_outcome": outcome(row),
                          "_variants": match_variants(row, feats.get("answer_text"))})

    try:
        plt = _plt()
    except ImportError:
        plt = None
    figs = _figures(plt, out, runs, focus, items, statuses) if plt else []

    summary = _summary(runs, focus, items, statuses)
    if plt is None:
        summary.append("No figures: matplotlib is not installed where this report was built "
                       "(pip install matplotlib).")
    html_path = out / "index.html"
    html_path.write_text(_html(runs_dir, runs, focus, items, figs, summary), encoding="utf-8")
    md_path = out / "report.md"
    md_path.write_text(_markdown(runs_dir, runs, focus, items, figs, summary), encoding="utf-8")
    return figs + [html_path, md_path]


def _figures(plt, out: Path, runs: list[dict], focus: Optional[dict], items: list[dict],
             statuses: Counter) -> list[Path]:
    """Draw every figure that has data; return the files written."""
    figs: list[Path] = []

    def add(p: Optional[Path]) -> None:
        if p is not None:
            figs.append(p)

    if focus:
        add(fig_scorecard(plt, focus, items, _arm_note(focus), out / "01_scorecard.png"))
        add(fig_training_curve(plt, focus, out / "02_training_curve.png"))
        add(fig_lr_grad(plt, focus, out / "03_learning_rate_and_gradients.png"))
        add(fig_pred_vs_gold(plt, items, out / "04_answer_vs_gold.png"))
        add(fig_outcomes(plt, items, out / "05_answer_outcomes.png"))
        add(fig_rules(plt, items, out / "06_score_by_matching_rule.png"))
        add(fig_error_size(plt, items, out / "07_error_size.png"))
        add(_breakdown(plt, items,
                       lambda i: (f"{i['_n_steps']} step" + ("s" if i["_n_steps"] > 1 else ""))
                       if i.get("_n_steps") else None,
                       "Accuracy by number of calculation steps",
                       "Steps in the gold program (e.g. subtract then divide = 2). More steps "
                       "= harder multi-step reasoning.", out / "08_accuracy_by_steps.png",
                       order=[f"{k} step" + ("s" if k > 1 else "") for k in range(1, 10)]))
        add(_breakdown(plt, [dict(i, _op=o) for i in items for o in (i.get("_ops") or [])],
                       lambda i: i["_op"].replace("_", " "), "Accuracy by operation",
                       "A question counts once for every operation its gold program uses.",
                       out / "09_accuracy_by_operation.png"))
        add(_breakdown(plt, items,
                       lambda i: i.get("_evidence") if i.get("_evidence") != "unknown" else None,
                       "Accuracy by where the evidence is",
                       "Whether the numbers needed come from the table, the text, or both.",
                       out / "10_accuracy_by_evidence.png",
                       order=["table only", "text only", "table + text"]))
        length_key, length_order = _length_buckets(items)
        add(_breakdown(plt, items, length_key, "Accuracy by prompt length",
                       "Questions split into four equal groups by prompt length (report "
                       "excerpt + question). Tests whether long reports hurt.",
                       out / "11_accuracy_by_prompt_length.png", order=length_order))
        add(fig_answer_length(plt, items, out / "12_reply_length.png"))
        add(fig_time_compute(plt, focus, out / "13_time_and_compute.png"))
    add(fig_all_runs(plt, runs, out / "14_all_experiments.png"))
    add(fig_frontier(plt, runs, out / "15_accuracy_vs_compute.png"))
    add(fig_gap(plt, runs, out / "16_contamination_gap.png"))
    add(fig_faithfulness(plt, runs, out / "17_faithfulness.png"))
    add(fig_status(plt, statuses, out / "18_registry_status.png"))
    return figs


def _roadmap_text() -> str:
    """Progress against docs/ROADMAP.md, or a note if it cannot be read."""
    try:
        from .roadmap import load, summary

        return summary(load())
    except Exception as e:  # noqa: BLE001 — the report must not fail on this
        return f"(roadmap not available: {e})"


def _arm_note(run: dict) -> str:
    try:
        from .config import load_yaml
        from .paths import configs_dir

        note = (load_yaml(configs_dir() / "arms.yaml").get(run["arm"]) or {}).get("notes", "")
    except Exception:  # noqa: BLE001
        note = ""
    parts = [f"{run['arm']}: {note}" if note else run["arm"], f"{run['model']}",
             f"{run['dataset']} {run['split']} split", f"seed {run.get('seed')}",
             f"code {run.get('git_commit', '?')}"]
    return "  |  ".join(parts)


def _summary(runs: list[dict], focus: Optional[dict], items: list[dict],
             statuses: Counter) -> list[str]:
    lines = [f"{len(runs)} completed experiment(s) in this folder; registry: " +
             (", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(statuses.items()))
              or "empty") + "."]
    if focus and items:
        n = len(items)
        counts = Counter(i["_outcome"] for i in items)
        best = max(RULES, key=lambda kr: sum(i["_variants"][kr[0]] for i in items))
        best_n = sum(i["_variants"][best[0]] for i in items)
        lines.append(f"Latest: {focus['name']} scored {_pct(focus.get('exact_match'))} exact "
                     f"match ({counts['Correct']} of {n}).")
        form = counts["Right value, written as a %"] + counts["Right arithmetic, not evaluated"]
        if form:
            lines.append(f"{form} answer(s) had the right value in the wrong form "
                         "(a percent sign or an unevaluated sum).")
        if counts["Close (within 1%)"]:
            lines.append(f"{counts['Close (within 1%)']} answer(s) were within 1% of gold "
                         "but not exact.")
        if best_n > counts["Correct"]:
            lines.append(f"Under the most lenient diagnostic rule the score would be "
                         f"{100 * best_n / n:.1f}%: the matching rule must be settled before "
                         "arms are compared.")
        tr = focus["_train"]
        if tr:
            lines.append(f"Training: {tr.get('n_train')} questions x {tr.get('epochs')} "
                         f"epoch(s), {tr.get('global_steps')} steps at "
                         f"{tr.get('tokens_per_s') or 0:,.0f} tokens/s.")
    return lines


def _results_rows(runs: list[dict]) -> list[list[str]]:
    rows = []
    for r in runs:
        tr = r["_train"]
        rows.append([r["name"], _pct(r.get("exact_match")), str(r.get("n_examples")),
                     _pct(r.get("execution_accuracy")), _pct(r.get("faithfulness_primary")),
                     f"{tr.get('train_loss'):.3f}" if tr.get("train_loss") is not None else "-",
                     f"{tr.get('tokens_per_s'):,.0f}" if tr.get("tokens_per_s") else "-",
                     _minutes(r.get("wall_time_s")), _si(r.get("train_flops")),
                     _si(r.get("infer_flops")), str(r.get("git_commit", ""))])
    return rows


_RESULT_HEAD = ["experiment", "exact match", "n", "exec acc", "faithfulness", "train loss",
                "tok/s", "wall", "train FLOPs", "infer FLOPs", "code"]
_ITEM_HEAD = ["#", "question", "gold", "human answer", "model reply", "outcome"]


def _item_rows(items: list[dict], limit: int = 80) -> list[list[str]]:
    rows = []
    for k, i in enumerate(items[:limit], 1):
        rows.append([str(k), (i.get("_question") or i.get("uid", ""))[:140], str(i["gold"]),
                     str(i.get("_answer_text") or ""), _first_line(i.get("text"))[:60],
                     i["_outcome"]])
    return rows


def _html(runs_dir: Path, runs: list[dict], focus: Optional[dict], items: list[dict],
          figs: list[Path], summary: list[str]) -> str:
    esc = html.escape

    def table(head: list[str], rows: list[list[str]]) -> str:
        th = "".join(f"<th>{esc(h)}</th>" for h in head)
        trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
        return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>"

    fig_html = "".join(
        f"<figure><img src='{esc(p.name)}' alt='{esc(CAPTIONS.get(p.name[:2], p.stem))}'>"
        f"<figcaption>{esc(p.name[:2])}. {esc(CAPTIONS.get(p.name[:2], ''))}</figcaption></figure>"
        for p in figs)
    counts = Counter(i["_outcome"] for i in items)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report</title>
<style>
:root {{ color-scheme: light; --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2};
  --muted:{MUTED}; --grid:{GRID}; }}
body {{ background:#f9f9f7; color:var(--ink); margin:0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.5; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; }} h2 {{ font-size: 1.1rem; margin-top: 2.2rem; }}
.sub {{ color: var(--ink2); margin: 0 0 16px; }}
.card {{ background: var(--surface); border: 1px solid rgba(11,11,11,.1); border-radius: 8px;
  padding: 12px 16px; }}
figure {{ margin: 20px 0; background: var(--surface); border: 1px solid rgba(11,11,11,.1);
  border-radius: 8px; padding: 8px; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ color: var(--ink2); font-size: .9rem; padding: 6px 4px 2px; }}
.tw {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-size: .85rem; width: 100%; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }} th {{ color: var(--ink2); font-weight: 600; }}
td {{ font-variant-numeric: tabular-nums; }}
pre {{ font-size: .8rem; overflow-x: auto; }}
</style></head><body><main>
<h1>Run report: {esc(runs_dir.name)}</h1>
<p class="sub">Generated {esc(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))} from
{esc(str(runs_dir))}. Figures are PNGs in this folder; the tables below hold every number.</p>
<div class="card"><ul>{''.join(f'<li>{esc(s)}</li>' for s in summary)}</ul></div>
<h2>Project roadmap</h2><pre class="card">{esc(_roadmap_text())}</pre>
<h2>Figures</h2>{fig_html or '<p>No completed experiment yet.</p>'}
<h2>All completed experiments</h2>{table(_RESULT_HEAD, _results_rows(runs))}
<h2>Answer outcomes{(' - ' + esc(focus['name'])) if focus else ''}</h2>
{table(['outcome', 'questions'], [[o, str(counts[o])] for o in OUTCOMES if counts[o]])}
<h2>Every answer (first {min(len(items), 80)})</h2>{table(_ITEM_HEAD, _item_rows(items))}
</main></body></html>
"""


def _markdown(runs_dir: Path, runs: list[dict], focus: Optional[dict], items: list[dict],
              figs: list[Path], summary: list[str]) -> str:
    def table(head: list[str], rows: list[list[str]]) -> str:
        clean = [[c.replace("|", "/").replace("\n", " ") for c in r] for r in rows]
        return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                         + ["| " + " | ".join(r) + " |" for r in clean])

    counts = Counter(i["_outcome"] for i in items)
    parts = [f"# Run report: {runs_dir.name}", "", *[f"- {s}" for s in summary], "",
             "## Figures", "", *[f"![{CAPTIONS.get(p.name[:2], p.stem)}]({p.name})" for p in figs],
             "", "## All completed experiments", "", table(_RESULT_HEAD, _results_rows(runs)), ""]
    if items:
        parts += [f"## Answer outcomes - {focus['name']}", "",
                  table(["outcome", "questions"], [[o, str(counts[o])] for o in OUTCOMES
                                                   if counts[o]]), "",
                  "## Matching rules (diagnostic)", "",
                  table(["rule", "exact match"],
                        [[name, _pct(sum(i["_variants"][k] for i in items) / len(items))]
                         for k, name in RULES]), "",
                  "## Every answer", "", table(_ITEM_HEAD, _item_rows(items, 1000)), ""]
    return "\n".join(parts)
