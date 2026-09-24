#!/usr/bin/env python
"""Build the contamination-controlled evaluation set (the "clean" split).

This is the linchpin artifact (proposal 5.3). IMPLEMENT the fetch/parse/generate
steps; the CLI, output contract, and the execution-verification gate are wired.

Pipeline:
  1. fetch post-cutoff 10-K/10-Q filings from SEC EDGAR (free public API),
     dated AFTER --cutoff (the max training cutoff across all models used)
  2. parse tables + surrounding text
  3. generate questions with deterministic numeric templates
     (pct change / ratio / sum / difference over extracted cells) -> gold value
  4. KEEP ONLY items whose gold program executes to the gold value (ror.sandbox)
  5. save as data/clean_set/clean.jsonl  (one Example-shaped record per line)

Output record schema (matches ror.data.Example):
  {uid, question, context, answer, gold_program, table_cells, meta}

Usage:
    python scripts/build_clean_set.py --cutoff 2025-01-01 --n 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.logging_utils import get_logger  # noqa: E402
from ror.sandbox import run_program  # noqa: E402

log = get_logger("ror.clean_set")


def fetch_filings(cutoff: str, max_filings: int) -> list[dict]:
    """Fetch filings dated after `cutoff` from SEC EDGAR. TODO: implement using
    the EDGAR full-text search / submissions JSON API. Respect EDGAR fair-access
    (a descriptive User-Agent, rate limiting)."""
    raise NotImplementedError("implement EDGAR fetch — see docstring")


def make_items(filings: list[dict], n: int) -> list[dict]:
    """Parse filings and generate templated numeric QA items with gold programs.
    TODO: implement table parsing + template question generation."""
    raise NotImplementedError("implement parsing + templated question generation")


def verify(items: list[dict]) -> list[dict]:
    """Keep only items whose gold_program executes to `answer`."""
    kept = []
    for it in items:
        prog = it.get("gold_program")
        if not prog:
            continue
        res = run_program(prog)
        if res.ok and _match(res.value, it["answer"]):
            kept.append(it)
    log.info("verified %d / %d items", len(kept), len(items))
    return kept


def _match(a, b) -> bool:
    from ror.metrics import answers_match
    return answers_match(a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="ISO date; keep filings AFTER this")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-filings", type=int, default=200)
    ap.add_argument("--out", default="data/clean_set/clean.jsonl")
    args = ap.parse_args()

    log.info("PLAN: fetch (>%s) -> parse -> generate -> verify -> save %s",
             args.cutoff, args.out)
    filings = fetch_filings(args.cutoff, args.max_filings)
    items = make_items(filings, args.n)
    items = verify(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    log.info("wrote %d clean items -> %s", len(items), out)


if __name__ == "__main__":
    main()
