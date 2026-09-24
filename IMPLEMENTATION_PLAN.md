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

## Phase 1 — Jobs 1–3 (free, on Kaggle)

### Data (`ror/data.py`)
- [ ] `load_dataset("finqa", split)` → list[Example] from Hugging Face
      (`ibm-research/finqa`); confirm the license first
- [ ] `linearize_table` + `normalize_numbers` preprocessing (helpers stubbed)
- [ ] `format_target(example, supervision)` for answer-only / CoT / PoT-gold
- [ ] Wire ConvFinQA + TAT-QA loaders (used in v2; keep interface identical)

### Contamination-controlled set (`scripts/build_clean_set.py`)
- [ ] Pull post-cutoff 10-K/10-Q filings from SEC EDGAR (record model cutoffs)
- [ ] Parse tables + text; generate template questions with deterministic gold
- [ ] Keep only items whose gold program executes (`ror.sandbox`)
- [ ] Save to `data/clean_set/` as the `clean` split; ~150–400 items
- [ ] **Check first:** did arXiv:2408.12337 release its GPT-4 traces? If so, reuse
      them and skip trace generation in Job 3.

### Models (`ror/models.py`)
- [ ] `load_student(name)` — 4-bit NF4 base + LoRA adapters (Qwen2.5-3B first)
- [ ] `load_teacher(name)` — Phase 1: ~32B on T4×2 (device_map); Phase 2: ≥70B
      via API/vLLM. Guard against loading ≥70B on free hardware.

### Training (`ror/training.py`) — Job 1
- [ ] QLoRA SFT loop (trl `SFTTrainer`), resumable checkpoints to `runs/<id>/`
- [ ] Get **one** arm end-to-end: A5, Qwen2.5-3B, FinQA — logged in results.jsonl
- [ ] Then A6, A7; then the size sweep (Qwen2.5-7B) and cross-family (Llama-3.1-8B)

### Inference + eval (`ror/inference.py`) — Job 2
- [ ] Greedy generation + batched decode; self-consistency (k configurable)
- [ ] Evaluate every arm on `standard` AND `clean`; log EM / faithfulness / exec
- [ ] Post-hoc program-induction proxy for non-PoT arms (`ror.faithfulness`)

### Large baseline + teacher (Job 3)
- [ ] A3/A4 with ~32B on T4×2
- [ ] Generate ~32B distilled PoT traces → verify → feed A8

### Aggregate
- [ ] `scripts/aggregate_results.py` → ablation table + contamination gap +
      accuracy-per-compute frontier from `runs/results.jsonl`
- [ ] Sanity pass: numbers reasonable, no missing arms, gaps computed

## Phase 2 — Job 4 (paid, only after Phase 1 is validated)

- [ ] Preemption re-check (has anyone published this exact study since?)
- [ ] `load_teacher` for ≥70B via open-70B API or rented A100-80GB (vLLM)
- [ ] Re-run A3/A4 and re-distill A8 at ≥70B; extend the frontier
- [ ] Final aggregation across both scales

## v2 backlog

- [ ] ConvFinQA (multi-turn) + TAT-QA; quantization-degradation curves;
      rolling clean set; more model families
