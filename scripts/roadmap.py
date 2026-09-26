#!/usr/bin/env python
"""Print progress against the approved roadmap (docs/ROADMAP.md).

Usage:
    python scripts/roadmap.py            # phases, current/parallel next tasks, GPU used
    python scripts/roadmap.py --next 5   # show more upcoming tasks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.roadmap import load, summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", type=int, default=3, help="tasks to list per open phase")
    ap.add_argument("--file", help="roadmap file (default docs/ROADMAP.md)")
    args = ap.parse_args()
    print(summary(load(args.file), next_n=args.next))


if __name__ == "__main__":
    main()
