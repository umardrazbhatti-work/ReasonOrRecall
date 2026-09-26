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
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.config import IDENTITY_FIELDS  # noqa: E402
from ror.results import load_results  # noqa: E402

METRICS = ["exact_match", "exact_match_strict", "execution_accuracy", "faithfulness_primary",
           "faithfulness_proxy", "executability_rate"]

# identity fields that are table axes (or averaged over) rather than a "variant"
_AXES = {"arm", "model", "dataset", "split", "seed", "role", "supervision",
         "inference", "phase"}


def _fmt(x) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _variant(row: dict) -> str:
    """Short id of the non-axis identity fields (hyperparameters, k, ...), so runs
    that differ in anything but the seed are never averaged together. Standard and
    clean runs of the same setup share a variant, which is what pairs them."""
    cfg = row.get("config") or {}
    if not cfg:
        return ""
    blob = json.dumps({k: cfg.get(k) for k in IDENTITY_FIELDS if k not in _AXES},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per exp_id — the latest. A --force re-run appends a second row for
    the same experiment; it must replace the old one, not count as another seed."""
    latest: dict[str, dict] = {}
    for r in rows:
        prev = latest.get(r["exp_id"])
        if prev is None or r.get("timestamp", 0) >= prev.get("timestamp", 0):
            latest[r["exp_id"]] = r
    return list(latest.values())


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse rows to one record per (dataset, arm, model, variant, split),
    averaging over seeds; then pair standard vs clean for the contamination gap."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in dedupe(rows):
        key = (r.get("dataset", "finqa"), r["arm"], r["model"], _variant(r), r["split"])
        buckets[key].append(r)

    per_split: dict[tuple, dict] = {}
    for (dataset, arm, model, variant, split), rs in buckets.items():
        rec = {"dataset": dataset, "arm": arm, "model": model, "variant": variant,
               "split": split, "n_seeds": len(rs),
               "n_examples": rs[0].get("n_examples", 0),
               "model_size_b": rs[0].get("model_size_b"),
               "train_flops": _avg([r.get("train_flops") for r in rs]),
               "infer_flops": _avg([r.get("infer_flops") for r in rs]),
               "cost_usd": _avg([r.get("cost_usd") for r in rs])}
        for m in METRICS:
            rec[m] = _avg([r.get(m) for r in rs])
        per_split[(dataset, arm, model, variant, split)] = rec

    # ablation table: one row per (dataset, arm, model, variant) — standard + clean + gap
    table: dict[tuple, dict] = {}
    for (dataset, arm, model, variant, split), rec in per_split.items():
        row = table.setdefault((dataset, arm, model, variant),
                               {"dataset": dataset, "arm": arm, "model": model,
                                "variant": variant})
        for m in METRICS:
            row[f"{m}_{split}"] = rec[m]
        row[f"n_{split}"] = rec["n_examples"]
        row[f"n_seeds_{split}"] = rec["n_seeds"]
    for row in table.values():
        em_std, em_cln = row.get("exact_match_standard"), row.get("exact_match_clean")
        row["contamination_gap"] = (
            round(em_std - em_cln, 4) if em_std is not None and em_cln is not None else None
        )

    frontier = [
        {"dataset": rec["dataset"], "arm": rec["arm"], "model": rec["model"],
         "variant": rec["variant"], "split": rec["split"],
         "model_size_b": rec["model_size_b"], "train_flops": rec["train_flops"],
         "infer_flops": rec["infer_flops"], "exact_match": rec["exact_match"],
         "cost_usd": rec["cost_usd"]}
        for rec in per_split.values()
    ]
    order = lambda r: (r["dataset"], r["arm"], r["model"], r["variant"])  # noqa: E731
    return sorted(table.values(), key=order), frontier


def _avg(vals):
    xs = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list({k for r in rows for k in r})
    lead = ["dataset", "arm", "model", "variant"]
    cols = sorted(cols, key=lambda c: (lead.index(c) if c in lead else len(lead), c))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    cols = ["dataset", "arm", "model", "variant", "exact_match_standard",
            "exact_match_clean", "contamination_gap", "faithfulness_primary_clean",
            "executability_rate_clean"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
