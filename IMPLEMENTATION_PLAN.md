# Implementation Plan

Work top to bottom. Tick a box only when the item is done **and** its test (or a
dry-run) passes. Keep public function signatures stable — `[IMPLEMENTED]` modules
depend on them. Update this file as you go; it is the shared plan.

## Phase 0 — Framework (mostly done in the scaffold)

- [x] Project structure, package layout, configs
- [x] `ror.config` — ExperimentConfig + deterministic `experiment_id`
- [x] `ror.registry` — status ledger, retry/skip policy (no repeat of finished work)
- [x] `ror.results` — RunResult schema + `runs/results.jsonl`
- [x] `ror.logging_utils`, `ror.utils` — logging, seeding, git commit, FLOP estimates
- [x] `ror.sandbox` — sandboxed Program-of-Thoughts executor
- [x] `ror.metrics`, `ror.faithfulness` — EM, execution accuracy, faithfulness
- [x] `ror.experiment` — orchestration shell (plan → train? → infer → eval → log)
- [x] `scripts/run_suite.py`, `run_experiment.py`, `aggregate_results.py`, `status.py`
- [x] Tests for registry / sandbox / metrics
- [ ] `pip install -e .`; run `pytest -q`; confirm all framework tests pass in Kaggle
      (passes locally, 76 tests; `notebooks/kaggle_runner.ipynb` runs it on Kaggle)
- [x] Code review of the scaffold: stub errors no longer burn registry attempts;
      results carry the full config; config identity/coercion fixes; metrics
      number parsing; sandbox blocks file/network/process access (proposal 5.4);
      git commit recorded from any cwd; aggregation dedupes re-runs and never
      mixes datasets/variants; suite filters `--split/--seed/--runs-dir`
- [x] Kaggle entry point `notebooks/kaggle_runner.ipynb` + `ror/kaggle.py`
      (clone at a ref, secrets, attached data, registry restore across versions)

## Phase 1 — Jobs 1–3 (free, on Kaggle)

### Data (`ror/data.py`)
- [x] `load_dataset("finqa", split)` → list[Example]. The HF repo `ibm-research/finqa`
      is only a loading script (unsupported by `datasets` ≥4), so the source is the
      original GitHub release pinned to a commit. License: MIT (FinQA, ConvFinQA),
      CC BY 4.0 (TAT-QA)
- [x] `scripts/prepare_data.py` → `dist/ror-data.zip` (Kaggle dataset `ror-data`):
      all gold programs converted to Python and execution-verified (FinQA and
      ConvFinQA 100%, TAT-QA arithmetic ~99%), sandbox spot-checked
- [x] `linearize_table` + `normalize_numbers` preprocessing
- [ ] `format_target(example, supervision)` for answer-only / CoT / PoT-gold
      (answer-only and PoT-gold done and tested; CoT (A6) needs a rationale
      format designed first)
- [x] `build_prompt` + `build_messages` for answer and PoT styles, few-shot,
      ConvFinQA history; shared by training and inference
- [x] Wire ConvFinQA + TAT-QA loaders (used in v2; keep interface identical)
      (TAT-QA span / multi-span scoring still needs v2 metrics)

### Contamination-controlled set (`scripts/build_clean_set.py`)
- [ ] Pull post-cutoff 10-K/10-Q filings from SEC EDGAR (record model cutoffs)
- [ ] Parse tables + text; generate template questions with deterministic gold
- [ ] Keep only items whose gold program executes (`ror.sandbox`)
- [ ] Save to `data/clean_set/` as the `clean` split; ~150–400 items
- [ ] **Check first:** did arXiv:2408.12337 release its GPT-4 traces? If so, reuse
      them and skip trace generation in Job 3.

### Models (`ror/models.py`)
- [x] `load_student(name)` — 4-bit NF4 (double quant, fp16 compute on T4) + LoRA
      adapter; CPU path tested, 4-bit path runs first in the smoke suite
- [ ] `load_teacher(name)` — Phase 1: ~32B on T4×2 (device_map); Phase 2: ≥70B
      via API/vLLM. (The ≥70B phase-1 guard and `resolve_model` are implemented
      and tested; loading is TODO.)

### Training (`ror/training.py`) — Job 1
- [x] QLoRA SFT loop (trl `SFTTrainer`), resumable checkpoints to `runs/<id>/`:
      completion-only loss (verified), checkpoints every 50 steps, clean pause
      before the session deadline (no attempt used), resume tested end-to-end
- [x] Smoke suite `configs/suite_smoke.yaml` (64 train / 32 eval, own runs dir)
- [ ] Smoke run passes on Kaggle T4; record tokens/s → decide epochs for the study
- [ ] Get **one** arm end-to-end: A5, Qwen2.5-3B, FinQA — logged in results.jsonl
- [ ] Then A6, A7; then the size sweep (Qwen2.5-7B) and cross-family (Llama-3.1-8B)

### Inference + eval (`ror/inference.py`) — Job 2
- [ ] Greedy generation + batched decode; self-consistency (k configurable)
      (greedy + batched done: explicit decoding config, no repetition penalty;
      per-item predictions.jsonl; self-consistency still TODO)
- [ ] Evaluate every arm on `standard` AND `clean`; log EM / faithfulness / exec
- [ ] Post-hoc program-induction proxy for non-PoT arms (`ror.faithfulness`)

### Large baseline + teacher (Job 3)
- [ ] A3/A4 with ~32B on T4×2
- [ ] Generate ~32B distilled PoT traces → verify → feed A8

### Aggregate
- [ ] `scripts/aggregate_results.py` → ablation table + contamination gap +
      accuracy-per-compute frontier from `runs/results.jsonl`
- [ ] Sanity pass: numbers reasonable, no missing arms, gaps computed

## Known issues to settle (before the runs they affect)

- **Percent scale (before A1–A4):** FinQA gold `exe_ans` stores "14%" as 0.14464,
  but `metrics.normalize_number("14%")` is 14.0, so prompted arms answering in
  percent are scored wrong. Decide the rule (e.g. accept x or x/100 for
  percentage questions) in `ror.metrics` before evaluating A1–A4. A5–A8 learn
  the decimal form and are unaffected.
- ~~**`max_seq_len`**~~ resolved: 2048 (decided 2026-09-25). Longer training
  items (~1.5%) are dropped, eval prompts are never truncated.
- **Compute budget (before the sweep):** 3 epochs at 2048 ≈ 20M training tokens
  per A5 run; the smoke run logs tokens/s — use it to fix epochs.
- **Session kills cost attempts:** a Kaggle session killed mid-run counts as an
  attempt (default `max_attempts: 2`). Mitigated: runs pause cleanly before
  `SESSION_HOURS` (no attempt used) and resume from the checkpoint.

## Phase 2 — Job 4 (paid, only after Phase 1 is validated)

- [ ] Preemption re-check (has anyone published this exact study since?)
- [ ] `load_teacher` for ≥70B via open-70B API or rented A100-80GB (vLLM)
- [ ] Re-run A3/A4 and re-distill A8 at ≥70B; extend the frontier
- [ ] Final aggregation across both scales

## v2 backlog

- [ ] ConvFinQA (multi-turn) + TAT-QA; quantization-degradation curves;
      rolling clean set; more model families
