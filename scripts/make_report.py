#!/usr/bin/env python
"""Write the figures + one-page report for a runs directory (see ror.report).

Usage:
    python scripts/make_report.py --runs-dir runs
    python scripts/make_report.py --runs-dir "Results/<folder>/runs_smoke"
Output: <runs-dir>/report/  (NN_*.png, index.html, report.md)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.report import build_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", help="output folder (default: <runs-dir>/report)")
    args = ap.parse_args()
    files = build_report(args.runs_dir, args.out)
    figs = [f for f in files if f.suffix == ".png"]
    print(f"report: {len(figs)} figures -> {files[-2].parent}")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
