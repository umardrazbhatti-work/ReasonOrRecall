# Reason or Recall?

Do small fine-tuned language models *reason* about financial numbers, or do they
*recall* a decade-old public benchmark? This repo runs a **contamination-controlled,
compute-matched** comparison of small QLoRA models vs. a large general model on
FinQA / ConvFinQA / TAT-QA. See `docs/proposal.docx` for the full proposal and
`CLAUDE.md` for how the code is meant to be built and run.

## Why the structure looks like this

Three requirements drove the design:

1. **Every run is logged** — full config + all metrics land in
   `runs/results.jsonl`. That file *is* the ablation study; `aggregate_results.py`
   turns it into the ablation table and the accuracy-per-compute frontier.
2. **Finished experiments never re-run** — `ror.registry` gives each experiment a
   deterministic id and tracks status. Completed and permanently-failed runs are
   skipped; a failed run retries up to `max_attempts`, then stops for good.
3. **Plan, then execute** — the suite runner prints a plan (and has `--dry-run`);
   each run logs its plan before doing anything.

## Workflow: code on GitHub, compute on Kaggle

1. Code changes are pushed to this repo.
2. `notebooks/kaggle_runner.ipynb` is uploaded to Kaggle once. Each run clones
   the repo at `GIT_REF`, installs it, and calls the scripts below. Setup steps
   (T4 x2, dataset, `HF_TOKEN` secret, resuming) are in its first cell.
3. The data is a Kaggle dataset built from `dist/ror-data.zip` (see Data).
4. Each committed notebook version saves `/kaggle/working/runs`. Attach it as an
   input to the next version to keep the registry, so finished experiments are
   never repeated.

## Data

```bash
python scripts/prepare_data.py   # -> data/ (gitignored) + dist/ror-data.zip
```

This downloads FinQA, ConvFinQA and TAT-QA from pinned upstream commits (MIT /
MIT / CC BY 4.0) and converts them to one `Example` schema in
`processed/<dataset>/<split>.jsonl`. Every gold program is converted to Python,
kept only if it re-executes to the gold answer, and spot-checked in the sandbox.
The zip also contains the raw files, a manifest (sha256s, counts, code commit)
and a dataset card. Upload it to Kaggle as a **private** dataset named `ror-data`.
Loaders read `$ROR_DATA_DIR` (default `data/`).

## Setup (local)

```bash
pip install -e .            # makes `ror` importable
pip install -r requirements.txt   # ML stack (in the Kaggle image)
pytest -q                   # tests should pass immediately
```

## Run

```bash
# see what the full Phase-1 sweep would do (no work done)
python scripts/run_suite.py configs/suite_phase1.yaml --dry-run

# start narrow: one arm, one model, one split, one seed, end-to-end
python scripts/run_suite.py configs/suite_phase1.yaml --only A5 --model qwen2.5-3b \
    --split standard --seed 0

# a single experiment from its config
python scripts/run_experiment.py configs/experiments/A5_qlora_answer.yaml

# where things stand
python scripts/status.py

# build the contamination-controlled eval set (implement first)
python scripts/build_clean_set.py --cutoff 2025-01-01 --n 300

# turn the logs into the ablation study
python scripts/aggregate_results.py
```

## Phased compute (free first)

| Job | What | Where | Cost |
|-----|------|-------|------|
| 1 | Student QLoRA fine-tuning (≤8B) | Kaggle T4 | Free |
| 2 | Student inference & evaluation | Kaggle T4 | Free |
| 3 | ~32B teacher traces + large baseline | Kaggle T4×2 | Free |
| 4 | **≥70B** scaling | API / rented A100 | ~$40–100 (Phase 2) |

**Job 4 is Phase 2 only.** A 70B model does not fit on free hardware; in Phase 1
the "large" model is ~32B on the dual-T4.

## What's implemented vs. TODO

Implemented: the framework (`config`, `registry`, `results`, `logging_utils`,
`utils`, `sandbox`, `metrics`, `faithfulness`, `experiment`, `paths`, `kaggle`),
the data layer (`data` loaders + `preprocess`), the model registry and ≥70B
guard in `models`, the Kaggle notebook, and all `scripts/` except the clean set.
Arm A5 (QLoRA answer-only) runs end to end: prompts/targets in `data`, 4-bit
student loading in `models`, resumable QLoRA in `training`, greedy batched
generation in `inference`. Run `configs/suite_smoke.yaml` first (about 20 min on a
T4), then `configs/suite_phase1.yaml`.
TODO (contracts in the docstrings): CoT targets (A6), teacher loading and
distilled traces (A3/A4/A8), self-consistency, the post-hoc faithfulness proxy,
and `scripts/build_clean_set.py`. Follow `docs/ROADMAP.md`. A run that
reaches an unimplemented stub returns to *pending* without using up an attempt.
