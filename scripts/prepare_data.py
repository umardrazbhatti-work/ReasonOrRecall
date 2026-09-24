#!/usr/bin/env python
"""Download the benchmarks from pinned upstream commits, preprocess them into
Example-shaped JSONL, verify gold programs, and package everything as one zip
to upload to Kaggle as a dataset.

Pipeline (logic lives in ror.preprocess; this script only orchestrates):
  1. download raw files -> <data-dir>/raw/<dataset>/   (skipped if present)
  2. build records      -> <data-dir>/processed/<dataset>/<split>.jsonl
  3. spot-check a sample of gold programs in the real sandbox (ror.sandbox)
  4. write <data-dir>/ror_data_manifest.json + README.md (dataset card)
  5. zip <data-dir> (manifest, card, processed/, raw/) -> dist/ror-data.zip

Usage:
    python scripts/prepare_data.py                       # all three datasets
    python scripts/prepare_data.py --datasets finqa --no-zip
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ror.data import STANDARD_SPLIT  # noqa: E402
from ror.logging_utils import get_logger  # noqa: E402
from ror.paths import MANIFEST_NAME  # noqa: E402
from ror.preprocess import (BUILDERS, PREPROCESS_VERSION, SOURCES,  # noqa: E402
                            download_source, gold_matches, sha256_file)
from ror.sandbox import run_program  # noqa: E402
from ror.utils import git_commit  # noqa: E402

log = get_logger("ror.prepare_data")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sandbox_spot_check(records: list[dict], n: int, seed: int = 0) -> dict:
    """Run a seeded sample of gold programs through ror.sandbox (the executor
    model programs use) and confirm they reproduce the gold answer there too."""
    with_prog = [r for r in records if r.get("gold_program")]
    sample = random.Random(seed).sample(with_prog, min(n, len(with_prog)))
    ok = 0
    for r in sample:
        res = run_program(r["gold_program"])
        tatqa = r["meta"].get("dataset") == "tatqa"
        ok += bool(res.ok and gold_matches(res.value, r["answer"], round_2dp=tatqa))
    return {"checked": len(sample), "agree": ok}


def dataset_card(manifest: dict) -> str:
    lines = [
        "# ror-data — benchmarks for *Reason or Recall?*",
        "",
        "Preprocessed FinQA, ConvFinQA and TAT-QA in one Example schema, plus the "
        "exact upstream raw files they were built from.",
        "Built by `scripts/prepare_data.py` in "
        "https://github.com/umardrazbhatti-work/ReasonOrRecall "
        f"(code commit `{manifest['code_commit'] or 'unknown'}`, "
        f"preprocess version {manifest['preprocess_version']}).",
        "",
        "## Layout",
        "",
        "```",
        f"{MANIFEST_NAME}          sources, sha256s, counts",
        "processed/<dataset>/<split>.jsonl  one Example per line",
        "raw/<dataset>/...                  upstream files, pinned commits",
        "clean_set/clean.jsonl              (added later: the post-cutoff set)",
        "```",
        "",
        "Record schema: `uid, question, context, answer, gold_program, table_cells, meta`.",
        "- `answer`: FinQA/ConvFinQA = executed value `exe_ans` (ratios as decimals, "
        "yes/no for comparisons); TAT-QA = gold answer in the table's units "
        "(`meta.scale`), multi-span answers joined with `; `.",
        "- `gold_program`: Python assigning `answer`, kept only when executing it "
        "reproduces the gold answer (`meta.program_verified`).",
        "- `table_cells`: table cells the gold program reads (for grounding checks).",
        "- ConvFinQA is turn-level; earlier turns and their answers are in `meta.history`.",
        "",
        "## Splits",
        "",
        "| dataset | split | n | verified gold programs |",
        "|---|---|---|---|",
    ]
    for ds, info in manifest["datasets"].items():
        for split, s in info["splits"].items():
            std = " (standard)" if STANDARD_SPLIT.get(ds) == split else ""
            lines.append(f"| {ds} | {split}{std} | {s['n']} | {s['verified_programs']} |")
    lines += ["", "## Sources and licenses", ""]
    for ds, info in manifest["datasets"].items():
        src = info["source"]
        lines.append(f"- **{ds}**: github.com/{src['repo']} @ `{src['commit'][:10]}` — "
                     f"{src['license']}. {src['citation']}")
    lines += ["", "Redistributed with attribution under the licenses above.", ""]
    return "\n".join(lines)


def make_zip(data_dir: Path, zip_path: Path) -> None:
    """Deterministic zip of the manifest, card, processed/ and raw/ (not clean_set
    unless it exists)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    members = [data_dir / MANIFEST_NAME, data_dir / "README.md"]
    for sub in ("processed", "raw", "clean_set"):
        if (data_dir / sub).exists():
            members += sorted(p for p in (data_dir / sub).rglob("*")
                              if p.is_file() and not p.name.endswith(".part"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in members:
            info = zipfile.ZipInfo(p.relative_to(data_dir).as_posix(),
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--zip", default=str(ROOT / "dist" / "ror-data.zip"))
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--spot-check", type=int, default=25,
                    help="gold programs per split to re-run in the sandbox")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    log.info("PLAN: download %s -> build processed JSONL -> sandbox spot-check -> "
             "manifest + card -> %s", ", ".join(args.datasets),
             "no zip" if args.no_zip else args.zip)

    manifest_file = data_dir / MANIFEST_NAME
    manifest = (json.loads(manifest_file.read_text(encoding="utf-8"))
                if manifest_file.exists() else {"datasets": {}})
    manifest.update({"name": "ror-data", "preprocess_version": PREPROCESS_VERSION,
                     "code_commit": git_commit(), "created_unix": int(time.time()),
                     "standard_split": STANDARD_SPLIT})

    for ds in args.datasets:
        src = SOURCES[ds]
        hashes = download_source(src, raw_dir, force=args.force_download, log=log.info)
        t0 = time.time()
        splits = BUILDERS[ds](raw_dir)
        info = {"source": {"repo": src.repo, "commit": src.commit, "license": src.license,
                           "citation": src.citation, "raw_sha256": hashes},
                "splits": {}}
        for split, records in splits.items():
            path = data_dir / "processed" / ds / f"{split}.jsonl"
            write_jsonl(path, records)
            check = sandbox_spot_check(records, args.spot_check)
            n_ver = sum(r["gold_program"] is not None for r in records)
            info["splits"][split] = {"n": len(records), "verified_programs": n_ver,
                                     "sandbox_spot_check": check,
                                     "sha256": sha256_file(path)}
            log.info("%s/%s: %d records, %d verified programs, sandbox %d/%d agree",
                     ds, split, len(records), n_ver, check["agree"], check["checked"])
            if check["agree"] != check["checked"]:
                raise SystemExit(f"sandbox disagrees with in-process execution on {ds}/{split}")
        manifest["datasets"][ds] = info
        log.info("%s built in %.1fs", ds, time.time() - t0)

    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (data_dir / "README.md").write_text(dataset_card(manifest), encoding="utf-8")
    log.info("wrote %s and README.md", manifest_file)

    if not args.no_zip:
        make_zip(data_dir, Path(args.zip))
        size = os.path.getsize(args.zip) / 1e6
        log.info("zip -> %s (%.1f MB) — upload it to Kaggle as a dataset", args.zip, size)


if __name__ == "__main__":
    main()
