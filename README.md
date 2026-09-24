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

## Setup

```bash
pip install -e .            # makes `ror` importable
pip install -r requirements.txt   # ML stack (in the Kaggle image)
pytest -q                   # framework tests should pass immediately
```

## Run

```bash
# see what the full Phase-1 sweep would do (no work done)
python scripts/run_suite.py configs/suite_phase1.yaml --dry-run

# start narrow: one arm, one model, end-to-end
python scripts/run_suite.py configs/suite_phase1.yaml --only A5 --model qwen2.5-3b

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

Implemented (framework): `config`, `registry`, `results`, `logging_utils`,
`utils`, `sandbox`, `metrics`, `faithfulness`, `experiment`, and all `scripts/`.
TODO (ML, contracts in the docstrings): `ror/data.py`, `ror/models.py`,
`ror/training.py`, `ror/inference.py`, and `scripts/build_clean_set.py`. Follow
`IMPLEMENTATION_PLAN.md`.
