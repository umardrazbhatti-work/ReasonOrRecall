# CLAUDE.md — Reason or Recall?

> This is the primary context file for Claude Code. **Read this file and
> `docs/ROADMAP.md` (the approved plan and tracker) in full before doing
> anything.** The full research
> proposal is in `docs/proposal.docx` (add it to the repo) — treat it as the
> source of truth for *why*; this file is the source of truth for *how*.

## 1. What this project is

An empirical study titled **"Reason or Recall?"**. We test whether small,
QLoRA-fine-tuned language models (≤8B) can match a large general model on
**financial numerical reasoning** (FinQA / ConvFinQA / TAT-QA) — and whether any
apparent parity reflects *genuine reasoning* or *memorization of a decade-old
public benchmark*. The two contributions are a **contamination-controlled**
evaluation (a held-out set built from filings created *after* model training
cutoffs) and a **compute-matched** frontier (accuracy per FLOP / dollar), plus a
**reason-vs-recall faithfulness** measure.

The "small model matches large" result alone is NOT novel (already shown by
arXiv:2408.12337). Our novelty is the *measurement protocol*. Keep that framing.

## 2. The four jobs (phased, free-first)

We validate the **entire** pipeline for free before spending anything.

| Job | Task | Hardware | Cost | Phase |
|-----|------|----------|------|-------|
| 1 | Student QLoRA fine-tuning (≤8B) | Kaggle T4 16 GB | Free | 1 |
| 2 | Student inference & evaluation | Kaggle T4 | Free | 1 |
| 3 | ~32B teacher-trace generation + large baseline | Kaggle **T4×2** (32 GB, 4-bit) | Free | 1 |
| 4 | **≥70B** teacher + baseline (scaling) | Open-70B API / rented A100-80GB | ~$40–100 | 2 |

**Hard rule: Job 4 is Phase 2 only.** A 70B model needs ~40 GB in 4-bit and does
**not** fit on free hardware. Do NOT attempt to load a ≥70B model on Kaggle. In
Phase 1 the "large" model is ~32B (e.g., Qwen2.5-32B) on the dual-T4.

## 3. Non-negotiable working rules

These exist because the human asked for them explicitly. Do not bypass them.

1. **Plan, then execute.** Before writing code for a new module or running any
   experiment suite, produce a short written plan (what, in what order, how you
   will verify) and proceed only after it is coherent. The suite runner
   (`scripts/run_suite.py`) prints a plan and supports `--dry-run`; use it.
2. **Log everything — the logs ARE the ablation study.** Every experiment MUST
   run through `ror.experiment.run_experiment` so its full config + all metrics
   land in `runs/results.jsonl`. Never run a one-off training/eval outside the
   runner — anything not logged there is lost from the final ablation. The
   record for each run captures every hyperparameter, the git commit, wall-time,
   estimated FLOPs, and every metric.
3. **Never repeat a finished experiment.** The registry (`ror.registry`) gives
   each experiment a deterministic id (hash of its config) and tracks status.
   `completed` and `permanently_failed` runs are skipped automatically. A
   `failed` run is retried only up to `max_attempts` (default 2), after which it
   becomes `permanently_failed` and is never retried. **Respect this** — do not
   pass `--force` to re-run finished work unless the human asks, and do not
   "helpfully" retry a permanently-failed experiment.
4. **Modular, typed, small.** One concern per module (see §5). Type hints and
   docstrings on every public function. Core logic lives in the `ror` package;
   Kaggle notebooks are thin entry points that import `ror` and call it — never
   put real logic in a notebook.
5. **Deterministic.** Seed everything (`ror.utils.set_seed`); record the seed and
   git commit in every result.
6. **Test before commit.** `pytest -q` must pass. Add a test when you add a
   module. The framework modules (registry, sandbox, metrics) already have tests.

## 4. Definition of done, per job

- **Job 1 done:** a student (start with Qwen2.5-3B) fine-tunes with QLoRA on
  FinQA under all four supervision arms (A5–A8) on a single T4; adapters saved
  under `runs/<exp_id>/`; runs logged.
- **Job 2 done:** every arm (A1–A8) evaluated on the **standard** FinQA test
  split AND the **clean** held-out set; EM, faithfulness, executability logged.
- **Job 3 done:** ~32B baseline (A3/A4) and ~32B-distilled traces (feeding A8)
  produced on Kaggle T4×2; contamination gap computable.
- **Job 4 done (Phase 2):** the large-model arms re-run with a ≥70B model on
  paid compute; frontier extended.
- **Study done:** `scripts/aggregate_results.py` turns `runs/results.jsonl` into
  the ablation table + the accuracy-per-compute frontier.

## 5. Repository map

```
ror/                     # the package — all real logic lives here
  config.py              # ExperimentConfig dataclass, YAML loader, deterministic experiment_id()
  registry.py            # experiment ledger: status + retry/skip policy  [IMPLEMENTED]
  results.py             # RunResult schema + append/load results.jsonl    [IMPLEMENTED]
  logging_utils.py       # console + per-run file logging                  [IMPLEMENTED]
  utils.py               # set_seed, git_commit, flop estimates            [IMPLEMENTED]
  sandbox.py             # sandboxed executor for Program-of-Thoughts code [IMPLEMENTED]
  metrics.py             # number normalization, Exact Match, exec accuracy [IMPLEMENTED]
  faithfulness.py        # primary (PoT) + post-hoc proxy faithfulness      [IMPLEMENTED]
  experiment.py          # orchestration: plan -> train? -> infer -> eval -> log [IMPLEMENTED shell]
  paths.py               # repo/data locations; $ROR_DATA_DIR                [IMPLEMENTED]
  data.py                # loaders, prompts, targets [IMPLEMENTED]; CoT target [TODO]
  preprocess.py          # pinned raw benchmarks -> verified Example JSONL    [IMPLEMENTED]
  kaggle.py              # Kaggle helpers: secrets, data dir, registry restore [IMPLEMENTED]
  models.py              # registry, ≥70B guard, 4-bit student load [IMPLEMENTED]; teacher load [TODO]
  training.py            # resumable QLoRA SFT with time budget [IMPLEMENTED]; distilled traces [TODO]
  inference.py           # greedy batched generation + parsing [IMPLEMENTED]; self-consistency, post-hoc proxy [TODO]
  report.py              # figures + one-page report per runs folder         [IMPLEMENTED]
configs/
  base.yaml              # shared defaults
  models.yaml            # model registry (students, teachers, sizes, cutoffs)
  arms.yaml              # A1..A8 supervision/inference specs
  suite_phase1.yaml      # the Phase-1 sweep (arms × students × seeds × splits)
  suite_smoke.yaml       # A5 on a small slice (own runs_smoke/ registry) — run first
scripts/
  run_suite.py           # expand a suite, skip finished, plan then run     [IMPLEMENTED]
  run_experiment.py      # run one experiment by config path                [IMPLEMENTED]
  prepare_data.py        # download + preprocess benchmarks -> dist/ror-data.zip [IMPLEMENTED]
  build_clean_set.py     # build the contamination-controlled eval set      [TODO — contract given]
  aggregate_results.py   # results.jsonl -> ablation table + frontier       [IMPLEMENTED]
  status.py              # registry dashboard                               [IMPLEMENTED]
  make_report.py         # <runs>/report/: NN_*.png + index.html            [IMPLEMENTED]
notebooks/
  kaggle_runner.ipynb    # the Kaggle entry point: clone -> install -> data -> restore -> test -> plan -> run
tests/                   # pytest; framework, data and Kaggle helpers covered
docs/                    # proposal.docx (= Proposal V2) + AUDIT_<date>.md (status vs proposal)
runs/                    # gitignored: per-run logs, adapters, results.jsonl
data/, dist/             # gitignored: processed data and the Kaggle data zip
Results/                 # gitignored: Kaggle run outputs, one folder per run (see its README)
```

Code lives on GitHub (umardrazbhatti-work/ReasonOrRecall) and runs on Kaggle
through `notebooks/kaggle_runner.ipynb`. On Kaggle the data comes from the
attached `ror-data` dataset (`$ROR_DATA_DIR`), secrets from Kaggle Secrets, and
the registry persists by attaching the previous notebook version's output.

`[IMPLEMENTED]` = working, do not rewrite without reason. `[TODO]` = your job;
the file contains a precise docstring contract and raises `NotImplementedError`.

## 6. How to work (suggested loop)

1. Read this file + `docs/ROADMAP.md`. Run `python scripts/roadmap.py` and
   `python scripts/status.py`.
2. Work only on the current phase's next task (by ID, e.g. P1.2); anything not
   in the roadmap needs an approved amendment (roadmap §6, §9). State a short
   plan.
3. Implement one `[TODO]` module. Keep the public signatures already defined —
   other code depends on them. Run `pytest -q`.
4. Dry-run the relevant suite: `python scripts/run_suite.py configs/suite_phase1.yaml --dry-run`.
5. Run for real. Confirm rows appear in `runs/results.jsonl`.
6. Tick the task in `docs/ROADMAP.md`, add the run log / decision log entry,
   commit with the task ID in the message (e.g. "P1.2: ...").

Start small: get **one** arm (A5, Qwen2.5-3B, FinQA, standard split) end-to-end
and logged before scaling to the full suite.

## 7. Environment notes

- Kaggle free tier: ~30 GPU-hours/week, ~9–12h sessions. **Checkpoint often**
  and make training resumable — a killed session must not lose a run.
- Use the **T4×2** accelerator, not P100 (P100 lacks the 4-bit NF4 kernels QLoRA
  needs). A ≤8B student trains and runs on `cuda:0` only: `ror.training` stops
  the HF Trainer from wrapping it in `DataParallel` across both T4s (that
  crashes with 4-bit weights and would double the effective batch). The second
  T4 is for the ~32B teacher (`device_map="auto"`).
- 4-bit QLoRA (NF4) via `bitsandbytes`; adapters via `peft`; SFT via `trl`.
- Resolve exact library versions in the Kaggle image and commit a lockfile;
  `requirements.txt` lists compatible ranges, not exact pins.
- Secrets (HF token, teacher API key) come from environment variables — see
  `.env.example`. Never commit real keys.
- Generated Program-of-Thoughts code is executed. `ror.sandbox` runs it in an
  isolated subprocess with a timeout and resource limits; do not weaken that.
