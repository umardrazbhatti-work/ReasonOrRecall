#!/usr/bin/env python
"""Turn runs/results.jsonl into the ablation study: a table of every arm × model
with metrics on standard vs. clean data, the contamination gap, and the points
for the accuracy-per-compute frontier.

Stdlib only (no pandas needed) so it always runs. Outputs:
    runs/ablation_table.csv
    runs/ablation_table.md
    runs/frontier.csv

Usage:
    python scripts/aggregate_results.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.results import load_results  # noqa: E402

METRICS = ["exact_match", "execution_accuracy", "faithfulness_primary",
           "faithfulness_proxy", "executability_rate"]


def _fmt(x) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse rows to one record per (arm, model, split), averaging over seeds;
    then pair standard vs clean to compute the contamination gap."""
    # average duplicate (arm, model, split) over seeds
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["arm"], r["model"], r["split"])].append(r)

    per_split: dict[tuple, dict] = {}
    for (arm, model, split), rs in buckets.items():
        rec = {"arm": arm, "model": model, "split": split,
               "n_seeds": len(rs), "n_examples": rs[0].get("n_examples", 0),
               "model_size_b": rs[0].get("model_size_b"),
               "infer_flops": _avg([r.get("infer_flops") for r in rs]),
               "cost_usd": _avg([r.get("cost_usd") for r in rs])}
        for m in METRICS:
            rec[m] = _avg([r.get(m) for r in rs])
        per_split[(arm, model, split)] = rec

    # ablation table: one row per (arm, model), standard + clean + gap
    table: dict[tuple, dict] = {}
    for (arm, model, split), rec in per_split.items():
        key = (arm, model)
        row = table.setdefault(key, {"arm": arm, "model": model})
        for m in METRICS:
            row[f"{m}_{split}"] = rec[m]
        row[f"n_{split}"] = rec["n_examples"]
    for key, row in table.items():
        em_std, em_cln = row.get("exact_match_standard"), row.get("exact_match_clean")
        row["contamination_gap"] = (
            round(em_std - em_cln, 4) if em_std is not None and em_cln is not None else None
        )

    frontier = [
        {"arm": rec["arm"], "model": rec["model"], "split": split,
         "model_size_b": rec["model_size_b"], "infer_flops": rec["infer_flops"],
         "exact_match": rec["exact_match"], "cost_usd": rec["cost_usd"]}
        for (arm, model, split), rec in per_split.items()
    ]
    return sorted(table.values(), key=lambda r: (r["arm"], r["model"])), frontier


def _avg(vals):
    xs = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list({k for r in rows for k in r})
    cols = sorted(cols, key=lambda c: (c != "arm", c != "model", c))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    cols = ["arm", "model", "exact_match_standard", "exact_match_clean",
            "contamination_gap", "faithfulness_primary_clean",
            "executability_rate_clean"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    rows = load_results(args.runs_dir)
    if not rows:
        print("no results yet in", Path(args.runs_dir) / "results.jsonl")
        return

    table, frontier = aggregate(rows)
    runs = Path(args.runs_dir)
    write_csv(runs / "ablation_table.csv", table)
    write_md(runs / "ablation_table.md", table)
    write_csv(runs / "frontier.csv", frontier)
    print(f"aggregated {len(rows)} run-records -> {len(table)} (arm,model) rows")
    print(f"  {runs/'ablation_table.md'}")
    print(f"  {runs/'ablation_table.csv'}")
    print(f"  {runs/'frontier.csv'}")


if __name__ == "__main__":
    main()
