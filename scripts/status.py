#!/usr/bin/env python
"""Print the experiment registry: counts by status and a per-run line. Use this
to see what is done, running, failed, or permanently failed before launching more.

Usage:
    python scripts/status.py
    python scripts/status.py --failed        # only failed / permanently-failed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.registry import Registry, Status  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--failed", action="store_true", help="show only failures")
    args = ap.parse_args()

    reg = Registry(args.runs_dir)
    states = reg.all_states()
    summary = reg.summary()

    print("=== registry summary ===")
    if not summary:
        print("  (no runs yet)")
    for status, n in sorted(summary.items()):
        print(f"  {status:20s} {n}")
    print(f"  {'total':20s} {len(states)}")

    if args.failed:
        states = [s for s in states
                  if s.status in (Status.FAILED.value, Status.PERMANENTLY_FAILED.value)]

    if states:
        print("\n=== runs ===")
        for s in sorted(states, key=lambda x: x.name):
            age = int(time.time() - s.last_update)
            line = f"  {s.status:20s} {s.exp_id}  {s.name}  (att {s.attempts}/{s.max_attempts}, {age}s ago)"
            if s.last_error:
                line += f"\n      last_error: {s.last_error.splitlines()[-1][:120]}"
            print(line)


if __name__ == "__main__":
    main()
