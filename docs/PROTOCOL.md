# Reason or Recall? — Pre-registered protocol

**Version 1.0 (2026-09-26). Status: awaiting approval (roadmap P1.9).**
Fixed before any study run. Any later change is a *deviation*: it is logged in
the decision log of `docs/ROADMAP.md` with its date and reason, and reported in
the paper. Implementation references are to the repository at the commit that
approves this file.

Why: the contribution is a measurement protocol (contamination-controlled,
compute-matched, reason vs recall), not a new method. A measurement paper is only
as credible as the guarantee that its analysis was not chosen after seeing the
results.

---

## 1. Questions and hypotheses (proposal §4)

- **RQ1** How large is the drop from standard-test accuracy to post-cutoff
  (clean) accuracy, per model class?
- **RQ2** At equal inference compute, where do small fine-tuned models sit
  relative to the large general model, on standard versus clean data?
- **RQ3** Does fine-tuning improve reasoning faithfulness more on clean items
  than on likely-seen ones?
- **RQ4** Does the program-distilled model's advantage over answer-only
  fine-tuning survive decontamination, in accuracy and faithfulness?

- **H1** Two contamination pathways: the large model's standard score is inflated
  by pretraining recall, the fine-tuned small model's by memorization of the
  training distribution; the large model shows the larger standard→clean drop.
- **H2** On clean data the compute-matched frontier shifts: the small-model
  advantage narrows or moves.
- **H3** Fine-tuning raises faithfulness more than accuracy, and this holds on
  clean items.
- **H4** Program distillation's advantage over answer-only fine-tuning partly
  survives decontamination but is smaller on clean data.

Model classes: **small-prompted** (A1, A2), **small-fine-tuned** (A5–A8),
**large** (A3, A4).

## 2. Data

| Split | Items | Use |
|---|---|---|
| FinQA train | 6,251 (items > 2,048 tokens with their target, ~1.5%, dropped) | training (A5–A8) |
| FinQA dev, first 200 | 200 | checkpoint selection and dev loss only |
| **standard** = FinQA test | 1,147 | evaluation (likely seen) |
| **clean** | ~400, built in roadmap P2 | evaluation (unseen by construction) |
| **control** | ~400, built in roadmap P2 | evaluation (style control) |

- Sources pinned to commits (FinQA czyssrs/FinQA@0f16e28, MIT); gold programs
  converted to Python and execution-verified (100% of FinQA).
- **Clean set:** 10-K filings for fiscal year 2025, filed on or after
  **2026-01-01**, from SEC EDGAR; tables + surrounding text in FinQA's format;
  questions from deterministic templates (percentage change, ratio, sum,
  difference, and 2–3-step compositions); every value used cross-checked against
  the filing's XBRL facts; every gold program executed; 50 random items checked by
  hand (≥95% correct). Stratified by template and number of steps.
- **Control set:** the same templates applied to FinQA **test** tables
  (pre-cutoff, likely seen). It isolates question style from recency:
  standard → control is the style effect, control → clean the
  recency/contamination effect.
- Model inputs: tables linearized to Markdown; thousands separators removed;
  nothing else rewritten (FinQA's gold programs use parenthesized numbers as
  positive magnitudes). Evaluation prompts are never truncated.

## 3. Models and cutoffs

| Role | Model | Cutoff used as the boundary |
|---|---|---|
| student (main) | Qwen2.5-3B-Instruct | 2024-09-19 (release date; cutoff unpublished) |
| student (scale) | Qwen2.5-7B-Instruct | 2024-09-19 (release date) |
| student (family) | Llama-3.1-8B-Instruct | 2023-12-31 (model card) |
| large + teacher, Tier 1 | Qwen2.5-32B-Instruct, 4-bit on 2×T4 | 2024-09-19 (release date) |
| large + teacher, Tier 2 | Qwen2.5-72B-Instruct, rented A100-80GB | 2024-09-19 (release date) |

The clean window (filed ≥ 2026-01-01) is more than 15 months after every
boundary. Release dates are upper bounds for unpublished cutoffs; contamination
can therefore be bounded, not proven absent (reported as such).

## 4. Arms

| Arm | Model | Training | Inference |
|---|---|---|---|
| A1 | small | none | zero-shot, answer instruction |
| A2 | small | none | few-shot: 3 fixed exemplars from FinQA train |
| A3 | large | none | zero-shot |
| A4 | large | none | few-shot, same exemplars as A2 |
| A5 | small | QLoRA, target = the answer | greedy |
| A6 | small | QLoRA, target = a rationale rendered deterministically from the gold program, then the answer | greedy |
| A7 | small | QLoRA, target = the gold program (Python) | greedy; program executed |
| A8 | small | QLoRA, target = the teacher's program, kept only if it executes to the gold answer | greedy; program executed |

- Answer instruction (all non-program arms): "Give only the final answer: a
  number (write ratios and percentages as decimals, e.g. 0.145 for 14.5%) or
  yes/no." Program arms: write Python assigning `answer`.
- Few-shot exemplars: chosen once, before any A2/A4 result, as the 3 shortest
  FinQA-train items that together cover a one-step, a two-step and a yes/no
  question; fixed for every model.
- A8 traces: the teacher answers every FinQA training question once in program
  form (one retry on failure); only programs that execute to the gold value are
  kept; the acceptance rate is reported. The arXiv:2408.12337 traces were not
  released (checked 2026-09-26), so A8 is "[4]-style" distillation with this
  study's hyperparameters and teacher, not an exact reproduction.

## 5. Training (A5–A8)

- QLoRA: 4-bit NF4, double quantization, fp16 compute (T4 has no bf16); LoRA on
  q/k/v/o/gate/up/down projections, r = 16, α = 32, dropout 0.05, adapter
  weights in fp32.
- Optimizer: paged AdamW 8-bit, learning rate 2e-4, cosine schedule with 3%
  linear warm-up; effective batch 16 (the batch/memory layout fixed by the
  roadmap P1.4 pilot); loss on the target tokens only.
- **1 epoch.** Checkpoints kept at 4 evenly spaced steps (the last included).
- **Checkpoint selection (the proposal's early stopping):** each kept checkpoint
  answers the 200 dev items with the evaluation code; the one with the highest
  primary dev EM is used; ties → higher dev faithfulness (program arms) → the
  earlier step.
- Decision rule (roadmap P3.2): if on the first A5 run the best checkpoint is the
  last one and dev EM is still rising by ≥ 2 points between the last two checks,
  a 2-epoch amendment is proposed before any other study run; otherwise 1 epoch
  stands for every run.
- Seeds: Qwen2.5-3B trained arms with seeds 0 and 1; 7B and 8B with seed 0;
  untrained arms are deterministic (greedy) and run once.
- Each adapter is trained once and evaluated on all three evaluation splits.

## 6. Inference

Greedy decoding (no sampling, repetition penalty 1.0), at most 512 new tokens,
batches of 8, no prompt truncation. Programs run in a sandboxed subprocess (no
file, network or process access; 5 s CPU, 512 MB memory).

## 7. Metrics

- **Exact match, primary rule** (reported as EM): an answer is correct if it
  matches the gold under the strict rule, or is within 1% of the gold value
  (relative; absolute floor 1e-4); an answer written with a percent sign is also
  tried as its decimal. Arithmetic written in an answer is **not** evaluated.
  Yes/no and text answers: normalized string match. Rationale: FinQA's own human
  answers are rounded (e.g. "14%" for 0.14464), and the rule is the same for
  every arm.
- **Exact match, strict rule** (reported alongside): within 0.001 (absolute
  below 1, relative above); "15%" read as 15.
- **Execution accuracy** (program arms): the executed value is correct (primary
  rule). **Executability:** share of programs that run to a value.
- **Faithfulness, primary** (A7, A8): among correct answers, the share produced
  by a program that executes to that answer; reported with and without the
  grounding check (the numbers the program uses appear in the gold table cells).
- **Faithfulness, post-hoc proxy** (all other arms): the model writes a program
  for its own earlier answer; share whose execution agrees with that answer.
  Always labelled as the weaker proxy; strong claims rest on the primary measure.
- **Contamination gap:** EM(standard) − EM(clean), per run and per model class.
  **Decomposition:** style = EM(standard) − EM(control); recency =
  EM(control) − EM(clean).
- **Compute:** inference FLOPs per question = 2 × parameters × (prompt +
  generated tokens); training FLOPs = 4 × parameters × training tokens (QLoRA:
  no weight gradients for the frozen base), × 1.33 when gradient checkpointing
  recomputes the forward pass (`ror.utils`). **Cost:** GPU-hours × $/GPU-hour
  (prices in `configs/pricing.yaml`, filled with source and date before the
  frontier is drawn).
- **Likely seen (for RQ3):** a standard item is "likely seen" by a model if its
  membership score (Min-K% token probability) exceeds the 95th percentile of the
  same score on clean items (known non-members), or if the model reproduces the
  item's question verbatim from its first half.

## 8. Statistical analysis

- 95% percentile-bootstrap intervals (10,000 resamples, seeded) over test items;
  with several seeds, each item's correctness is averaged over seeds first and
  the seed-to-seed SD reported separately.
- Contamination gap: standard and clean items resampled separately.
- Arm vs arm on the same items: paired bootstrap interval and exact McNemar test.
- Difference of gaps (e.g. large vs small, H1): items resampled within each
  split, paired across the two systems on the same items.
- Families corrected with Holm-Bonferroni at α = 0.05: {H1 tests}, {H2}, {H3},
  {H4} (the planned tests below). Effect sizes with intervals are reported for
  every comparison, significant or not; a non-significant result is reported as
  such, with its interval, and never dropped.

### Planned tests (confirmatory)
1. **H1-a** gap(large) − gap(small-fine-tuned) > 0, per tier (32B; 72B in Tier 2),
   using the best arm per class on standard (named in advance: A4 for large, A8
   for small-fine-tuned), per student model.
2. **H1-b** fine-tuning pathway: [EM gain of A5 over A1 on standard items whose
   company-year is in FinQA train] − [same gain on items whose company-year is
   not] > 0.
3. **H2** [A8 − A4] on clean < [A8 − A4] on standard, at the main student (3B)
   and at each scale.
4. **H3** for each small model: [A7 − A1 in faithfulness] > [A7 − A1 in EM], on
   clean items; faithfulness measured with the post-hoc proxy for both arms (like
   for like), the primary measure for A7 reported alongside.
5. **H4** [A8 − A5] on clean > 0 and [A8 − A5] on clean < [A8 − A5] on standard.

Everything else (per-operation breakdowns, error types, other arm pairs) is
exploratory and labelled so.

## 9. Deviations and reporting

- Changes after approval go into the roadmap's decision log (date, what, why)
  and a "deviations from the protocol" paragraph in the paper.
- Runs are never repeated to get a better number: the registry skips completed
  and permanently failed experiments; a failed run is retried at most once.
- Every number in the paper is generated from the frozen results tags
  (`results-tier1`, `results-tier2`) by the repository's scripts.

## 10. Stated in advance as limitations

Release dates bound unpublished cutoffs; membership inference is noisy; the
post-hoc proxy can rationalize; the clean set covers FY2025 10-Ks only and ~400
items (gap resolution ≈ ±6 points); A8 uses a 32B/72B teacher, not GPT-4; one
seed above 3B.
