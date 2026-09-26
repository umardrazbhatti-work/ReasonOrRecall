# Reason or Recall? — Roadmap (phases, tasks, gates)

**Status: APPROVED v1.0 (2026-09-26).** This file is the single tracker for the
project (it replaced IMPLEMENTATION_PLAN.md) and is followed strictly: work
happens only on tasks listed here; any change goes through an amendment the
project owner approves (working rules, section 6; decision log, section 9).

Source of truth for *why*: `docs/proposal.docx` (= Proposal V2, 22-9-26).
Audit that motivated this plan: `docs/AUDIT_2026-09-26.md`.

**Naming.** Roadmap phases are **P0–P10**. The proposal's funding split is called
**Tier 1** (free, Kaggle, ~32B large model) and **Tier 2** (paid, ≥70B) here, to
avoid two different "Phase 1"s. (In code, `phase: 1|2` means Tier 1|2.)

Status marks: `[x]` done · `[~]` in progress / partly done · `[ ]` not started.
Priority marks (used where GPU quota could run short): **M** must (the paper
fails without it) · **S** should · **C** could.

---

## 1. The phases at a glance

| Phase | Goal | GPU (est.) | Status |
|---|---|---|---|
| **P0** Foundations | Pipeline runs end to end on Kaggle; data; logging; reports | ~0.3 h used | **Done** (G0 2026-09-26) |
| **P1** Protocol freeze | Settle everything that changes results or ids *before* the first study run | ~3 h | Partly done |
| **P2** Clean evaluation set | The key artifact: ~400 post-cutoff EDGAR items + ~400 style-control items | 0 | Not started |
| **P3** Main student, Qwen2.5-3B | All student arms, 2 seeds for trained arms | ~64 h | Not started |
| **P4** Large model, ~32B (Tier 1) | A3/A4 baselines; PoT traces; A8 on 3B | ~38 h | Not started |
| **P5** Scale and family | Qwen2.5-7B, Llama-3.1-8B on the key arms | ~124 h | Not started |
| **P6** Reason vs recall | Grounding, post-hoc proxy, membership inference, train-overlap | ~24 h | Not started |
| **P7** Analysis | RQ1–RQ4 / H1–H4 with confidence intervals; results frozen | 0 | Not started |
| **P8** Tier 2 (paid) | ≥70B baseline + teacher; two-scale frontier | ~15–20 h rented A100 | Not started |
| **P9** Paper and release | Draft, figures, released set/code/adapters, submission | 0 | Not started |
| **P10** v2 extension | ConvFinQA + TAT-QA (the proposal's "later iteration") | later | After submission |

Tier-1 GPU total ≈ **250 Kaggle GPU-hours** ≈ 8–9 weeks of the ~30 h/week quota
(basis: measured 310 training tokens/s for 3B; 7B/8B/32B scaled, to be measured in
their pilots). See section 5.

**Where we are (2026-09-26):** P0 done except the tracker script; first
end-to-end Kaggle run completed; P1 has its diagnostics but no decisions
implemented; nothing of P2–P9 started. Roughly **10–15% of the total work**.

**Critical path:** P1 → P2 (clean set) → every "clean" number. P2 needs no GPU and
runs in parallel with P3–P5.

---

## 2. Phases in detail

Each task has an ID (cite it in commits and messages), a deliverable, and "done
when". A phase ends only when its **gate** passes; gate evidence is recorded in
the run log (section 6).

### P0 — Foundations  *(done; G0 passed 2026-09-26)*
- [x] **P0.1** Framework: config + deterministic experiment ids, registry (never repeat finished work), results log, sandboxed program executor, metrics, primary faithfulness.
- [x] **P0.2** Data: FinQA / ConvFinQA / TAT-QA from pinned sources; gold programs converted to Python and execution-verified (FinQA, ConvFinQA 100%); licenses recorded; `ror-data.zip`.
- [x] **P0.3** Kaggle runner: clone at a commit, secrets, data, registry restore, pause/resume before the 12 h limit, verdict line, one output zip.
- [x] **P0.4** 4-bit QLoRA training (resumable) and greedy batched inference on the T4, with the three Kaggle-only fixes (torchao, DataParallel, bf16).
- [x] **P0.5** First end-to-end run on Kaggle (smoke, 2026-09-26): 310 tok/s training, ~1 answer/s.
- [x] **P0.6** Run report: 10–18 figures + `index.html` per runs folder, inline in the notebook, in the zip.
- [x] **P0.7** Tracker: `scripts/roadmap.py` prints phase progress, next tasks and GPU-hours used vs budget (from this file + the run log); shown at the top of every Kaggle session and report.

**Gate G0:** P0.7 done; `python scripts/roadmap.py` matches this file.

### P1 — Protocol freeze  *(no study run starts before G1)*
Everything here changes metrics or experiment ids, so it must be settled first.
- [x] **P1.1 (M) Train once, evaluate on every split.** Today a trained arm is retrained for each test split (the split is part of the experiment id), which would double or triple training compute. Adapters get their own id from the *training* fields only; evaluation experiments (standard / clean / control) reuse them. Done when: an A5 run on two splits trains once (test proves it).
- [x] **P1.2 (M) Answer-matching rule** (amendment M4): primary EM = numeric match within 1% relative, with a percent-form answer read as its decimal; strict EM reported alongside; arithmetic written in an answer is never evaluated. Both stored per run; the smoke run re-scored (primary 6.2%, strict 0.0%). Done when: `ror.metrics` + tests + both numbers in results and report.
- [x] **P1.3 (M) Checkpoint selection = the proposal's early stopping** (amendment M2): 1 epoch; dev evaluation on 200 fixed dev items at 4 evenly spaced checkpoints; keep the best by dev EM (tie-break: faithfulness for program arms); curve logged. Done when: tested on the tiny model; the dev curve appears in the report.
- [ ] **P1.4 (M) Throughput pilot** on Kaggle (≤3 GPU-h): batch 1×16 vs 4×4 grouped by length, gradient checkpointing on/off for 3B; pick the fastest that fits memory. Effective batch stays 16. Done when: chosen setting in `configs/base.yaml`, measured tok/s recorded.
- [x] **P1.5 (M) Statistics** (`ror/stats.py`): 95% bootstrap CIs for EM, gaps and faithfulness; paired bootstrap / McNemar for arm-vs-arm on the same items; Holm correction per hypothesis family. In aggregation and report. Done when: tested; CIs visible in the report.
- [ ] **P1.6 (S) Dollar convention** (`configs/pricing.yaml`): $/GPU-hour per hardware (source + date) and API $/token → `cost_usd` per run → per-dollar frontier. Done when: nonzero `cost_usd` in a test run.
- [ ] **P1.7 (M) Final experiment matrix** as suites per phase (`suite_p3_3b.yaml`, `suite_p4_32b.yaml`, `suite_p5_scale.yaml`) with the priorities of section 4; the dry-run prints the plan and its GPU-hour estimate.
- [ ] **P1.8 (M) Desk check:** did arXiv:2408.12337 [4] release its traces, and under what license? Decides P4.3 (reuse vs generate).
- [ ] **P1.9 (M) `docs/PROTOCOL.md`**: a pre-registration of RQs, hypotheses, splits, metrics, matching rule, selection rule, seeds, statistical tests and the decision rules of this roadmap. Approved by you (ideally shown to your supervisor).
- [ ] **P1.10 (M) GPU check:** one smoke run on Kaggle with all P1 changes (~10 min).

**Gate G1:** P1.1–P1.10 done; all tests pass; the Kaggle smoke run passes with the new protocol; PROTOCOL.md approved.

### P2 — Clean evaluation set  *(no GPU; starts with P1, runs in parallel)*
- [ ] **P2.1 (M)** EDGAR fetcher: 10-K filings of fiscal year 2025, **filed on or after 2026-01-01** (≥15 months after the latest Tier-1 cutoff proxy, 2024-09-19); a fixed, logged list of accession numbers; cached; SEC fair-access (descriptive User-Agent with a contact you provide, ≤10 requests/s).
- [ ] **P2.2 (M)** Table + surrounding-text extraction from the filing HTML into FinQA's format (pre-text, table, post-text).
- [ ] **P2.3 (M)** Every table value used by a question is cross-checked against the filing's XBRL facts (addition A3); mismatches are dropped.
- [ ] **P2.4 (M)** Deterministic templates: percentage change, ratio, sum, difference, plus 2–3-step compositions so the step mix resembles FinQA's (57% one step, 36% two steps). Gold = FinQA-style program.
- [ ] **P2.5 (M)** Execution gate (already built): keep only items whose program executes to the gold value.
- [ ] **P2.6 (M)** **~400 clean items**, stratified by template and step count (400 = the proposal's upper bound, chosen for statistical power: section 5).
- [ ] **P2.7 (M)** **Style control split (~400 items, addition A2)**: the same templates on FinQA **test** tables (pre-cutoff, likely seen). Lets the paper split the standard→clean drop into a question-style effect and a recency/contamination effect.
- [ ] **P2.8 (M)** Manual spot-check (proposal §7): a review sheet of 50 random items; **you** check them; ≥95% correct or fix and repeat.
- [ ] **P2.9 (M)** Package `ror-data` v2 (clean + control), manifest, datasheet (sources, window, templates, licenses and EDGAR terms). You upload it to Kaggle.

**Gate G2:** ~400 clean + ~400 control items verified; spot-check ≥95%; datasheet written; Kaggle dataset updated.

### P3 — Main student: Qwen2.5-3B  *(Jobs 1–2; ~64 GPU-h)*
Every trained run: train once (1 epoch, best of 4 dev checks), then evaluate on standard, clean and control.
- [ ] **P3.1 (M)** A5, seed 0: first study run.
- [ ] **P3.2 (M) Epoch gate:** if P3.1's best checkpoint is the last one *and* dev EM is still rising by ≥2 points, propose an amendment to 2 epochs (budget permitting); else 1 epoch stands for all runs.
- [ ] **P3.3 (M)** A7 (PoT gold), seed 0.
- [ ] **P3.4 (M)** A6: rationales rendered deterministically from the gold program (amendment M8), then A6 seed 0.
- [ ] **P3.5 (M)** A1 (zero-shot) and A2 (few-shot; 3 fixed, short exemplars from train, covering different operations).
- [ ] **P3.6 (M)** Seed 1 for A5, A6, A7 (proposal §7: a small seed sweep for LoRA variance).
- [ ] **P3.7 (M)** Clean and control evaluations for all of the above once G2 passes (evaluation only; no retraining).

**Gate G3:** A1, A2, A5, A6, A7 on 3B complete on all three splits, with reports; GPU spend within +20% of estimate (else re-plan before P4).

### P4 — Large model, Tier 1: ~32B  *(Job 3; ~38 GPU-h)*
- [ ] **P4.1 (M)** `load_teacher`: Qwen2.5-32B-Instruct, 4-bit, split across the two T4s; pilot measures memory and tokens/s; budget updated.
- [ ] **P4.2 (M)** A3 (zero-shot) and A4 (few-shot, same exemplars as A2) on all three splits.
- [ ] **P4.3 (M)** PoT traces on FinQA train: reuse [4]'s if P1.8 allows (the proposal's mitigation), else 32B traces with 1 sample per item + 1 retry; keep only traces that execute to gold; record the acceptance rate. Capped by budget (proposal §7).
- [ ] **P4.4 (M)** A8 on 3B, seeds 0 and 1, evaluated on all three splits.

**Gate G4:** A3, A4, A8-3B complete on all splits; trace statistics logged.

### P5 — Scale sweep and cross-family  *(~124 GPU-h)*
- [ ] **P5.1 (M)** Pilot for Qwen2.5-7B and Llama-3.1-8B (memory, tokens/s); budget re-estimated; HF_TOKEN access to Llama confirmed.
- [ ] **P5.2 (M)** 7B and 8B: A1, A5, A8 (seed 0), all splits.
- [ ] **P5.3 (S)** 7B and 8B: A2, A7.
- [ ] **P5.4 (C)** 7B and 8B: A6; optional students (Phi-4-mini, SmolLM3) on A1/A5.

**Gate G5:** all **M** runs complete; S/C runs done or explicitly deferred in the decision log.

### P6 — Reason vs recall measures  *(~24 GPU-h)*
- [ ] **P6.1 (M)** Grounding check (proposal §5.5): a program counts as grounded if the numbers it uses come from the gold table cells; faithfulness is reported with and without it. No GPU.
- [ ] **P6.2 (M)** Post-hoc program-induction proxy for every non-program arm (A1–A6, A3/A4): the model writes a program for its own earlier answer; executed; agreement reported, labelled as a weaker proxy.
- [ ] **P6.3 (M)** Membership inference / verbatim reproduction (proposal §7) on standard items for each base model (3B, 7B, 8B, 32B): Min-K% token probability plus a verbatim-continuation test. **Clean items serve as known non-members** to calibrate the threshold (addition A5). Output: a per-item "likely seen" label (needed by RQ3).
- [ ] **P6.4 (M)** Fine-tuning pathway (H1's second pathway, addition A4): test items from a company-year that is also in FinQA train (1,044) vs not (103); clean items from companies in FinQA train vs new companies. No GPU.
- [ ] **P6.5 (M)** Per-item labels joined into one table: seen / unseen / clean, company overlap, grounded, faithful.

**Gate G6:** labels available for every test item and every run.

### P7 — Analysis and results freeze  *(no GPU)*
- [ ] **P7.1 (M)** RQ1–RQ4 and H1–H4 answered with 95% CIs and the tests in PROTOCOL.md.
- [ ] **P7.2 (M)** Gap decomposition: standard → control (question style) → clean (recency/contamination), per model class.
- [ ] **P7.3 (M)** Compute-matched frontier (FLOP and dollar), standard vs clean.
- [ ] **P7.4 (M)** Error analysis and sanity pass: no missing arm, every number traceable to a run.
- [ ] **P7.5 (M)** Results frozen: git tag `results-tier1`; the paper's tables and figures generated from it.

**Gate G7:** every RQ has a table or figure with CIs; tag created.

### P8 — Tier 2 (paid, ≥70B)  *(you approve the spend before it starts)*
- [ ] **P8.1 (M)** Preemption re-check (has anyone published this study?) → go / no-go for spending.
- [ ] **P8.2 (M)** Qwen2.5-72B-Instruct on a rented A100-80GB with vLLM, ~15–20 h (amendment M6): A3, A4 on all splits; membership inference; post-hoc proxy; traces.
- [ ] **P8.3 (M)** A8 trained on 72B traces for 3B (Kaggle, free); **S**: also 7B.
- [ ] **P8.4 (M)** Two-scale frontier and tier comparison.

**Gate G8:** Tier-2 rows in the frozen results (tag `results-tier2`).

### P9 — Paper and release
- [ ] **P9.1 (M)** Outline + venue plan: primary the ACL Rolling Review cycle of **February 2027** (ACL 2027 / Findings); fallback a *CL workshop on finance NLP or data contamination, or ICAIF 2027. Deadlines to be confirmed when chosen.
- [ ] **P9.2 (M)** Final figures and tables from the frozen results.
- [ ] **P9.3 (M)** Draft: intro (measurement, not method), related work [1]–[17] plus anything newer, method, clean-set datasheet, results RQ1–RQ4, limitations (temporal cutoff is a bound; proxy is weaker), ethics and licensing.
- [ ] **P9.4 (M)** Release: clean + control sets (license notes, EDGAR terms), code at a tag, adapters on the HF Hub.
- [ ] **P9.5 (M)** Preemption re-check at draft freeze (proposal §7); supervisor review; arXiv; submission.

**Gate G9:** submitted and on arXiv.

### P10 — v2 extension  *(after submission; the proposal's "later iteration")*
- [ ] **P10.1** ConvFinQA (dev as anchor) and TAT-QA runs with their own metrics; quantization curves; a rolling clean set.

---

## 3. Proposal coverage (nothing missed)

| Proposal item | Task(s) |
|---|---|
| §1/§3 framing: measurement, not method | P9.3 |
| RQ1 contamination gap per model class | P2, P3–P5, P7.1, P7.2 |
| RQ2 frontier at equal inference compute, standard vs clean | P1.6, P7.3, P8.4 |
| RQ3 faithfulness gain, clean vs likely-seen | P6.1–P6.3, P6.5, P7.1 |
| RQ4 distillation advantage survives decontamination | P4.4, P5.2, P7.1 |
| H1 two pathways | P6.3 (pretraining recall), P6.4 (training distribution), P7.2 |
| H2, H3, H4 | P7.3, P7.1 |
| §5.1 FinQA (v1); ConvFinQA/TAT-QA later; licenses | P0.2, P10.1, P9.4 |
| §5.2 students 3B/7B/8B (+ optional Phi-4-mini, SmolLM3); ~32B; ≥70B; cutoffs | P3, P5, P5.4, P4, P8, M7 |
| §5.3 clean set: EDGAR, parsing, templates, execution check, 150–400 | P2.1–P2.6 |
| §5.4 linearization, normalization | P0.2, M5 |
| §5.4 verified teacher traces | P4.3, P8.2 |
| §5.4 QLoRA per supervision format | P3, P4.4, P5 |
| §5.4 validate each epoch + early stopping | P1.3 (M2) |
| §5.4 error analysis and iterate | P0.6, P7.4 |
| §5.4 sandboxed execution | P0.1 |
| §5.5 primary faithfulness + grounding; post-hoc proxy | P0.1, P6.1, P6.2 |
| §5.6 arms A1–A8 on standard and clean; A3/A4/A8 per tier | P3, P4, P5, P8 |
| §5.6 metrics: EM, gap, execution accuracy, faithfulness, executability, FLOP + dollar frontier | P1.2, P1.5, P1.6, P6, P7 |
| §6 Jobs 1–4, T4×2 not P100, Tier 2 only after Tier 1 | P3/P5 (Jobs 1–2), P4 (Job 3), P8 (Job 4), gates |
| §7 32B throughput: cap samples, reuse [4], Colab | P4.3, P1.8, lever in section 5 |
| §7 preemption re-check | P8.1, P9.5 |
| §7 contamination as a bound + membership inference | P6.3, P9.3 |
| §7 proxy is weaker | P6.2, P9.3 |
| §7 clean set verified + spot-checked | P2.3, P2.5, P2.8 |
| §7 executability as a finding; seed sweep | P7.1, P3.6 |
| §7 licenses and EDGAR terms | P2.1, P9.4 |
| §8 contributions 1–4 (incl. released set, code, adapters) | P7, P8, P9.4 |

---

## 4. Amendments and additions (for your approval)

### Amendments: where the proposal has to change, and why
| ID | Change | Why | Effect on the paper |
|---|---|---|---|
| **M1** | Compute and timeline: Jobs 1–2 ≈ 210 GPU-h (was 120–180); Tier 1 ≈ 250 GPU-h ≈ 8–9 weeks of quota; overall ~16 weeks to submission (was 8–12) | Measured: 310 tok/s, ~6 h per FinQA epoch for 3B | None on the science; later date |
| **M2** | 1 epoch with 4 dev checkpoints, best kept by dev EM (tie-break faithfulness), instead of "validate each epoch and early-stop" | 3 epochs × all runs is 4× over budget; this keeps selection by dev accuracy and faithfulness | Same principle, cheaper; P3.2 gate can raise it to 2 epochs |
| **M3** | Full arm set on 3B (2 seeds for trained arms); 7B and 8B run A1, A5, A8 (must), A2, A7 (should), A6 (could), 1 seed | The proposal's matrix is by model class; the scale sweep and family check need the arms that answer RQ1–RQ4 | Keeps every RQ at every scale; A6 at 7B/8B only if budget allows |
| **M4** | EM = within 1% relative, percent form accepted; strict EM reported too; no evaluation of arithmetic in answers | Proposal leaves matching undefined; strict matching penalises answer-only arms and would inflate H4; FinQA's own human answers are rounded ("14%" for 0.14464) | Fair across arms; both numbers published |
| **M5** | Number normalization (thousands separators, scale words, parenthesized negatives, percentages) applied when **scoring**; model **inputs** only lose thousands separators | FinQA's gold programs use parenthesized numbers as positive magnitudes; rewriting inputs would contradict the gold | Same answers, no corrupted inputs |
| **M6** | Tier 2 = Qwen2.5-72B-Instruct on a rented A100-80GB (not a frontier API) | Same family as the 32B (clean scale step); cutoff before the clean window; logprobs needed for membership inference | Proposal allowed either; this one keeps the clean set valid |
| **M7** | Qwen2.5 cutoffs (unpublished) use the release date 2024-09-19 as an upper bound; clean filings are filed on or after 2026-01-01 | Remove doubt about the boundary | Stated in the paper |
| **M8** | A6 rationales are rendered deterministically from the gold program into natural language | Proposal doesn't say where rationales come from; teacher-written ones cost 32B hours | A6 and A7 then carry the same reasoning in different forms (text vs code): a cleaner comparison |

### Additions: small, in-theme, and they raise acceptance odds
| ID | Addition | Why it helps the paper |
|---|---|---|
| **A1** | Pre-registered protocol (P1.9) | Reviewers trust a measurement paper whose analysis was fixed before the results |
| **A2** | Style control split (P2.7) | Answers the obvious reviewer objection that "clean is just a different question style"; turns one gap into two measured effects |
| **A3** | XBRL cross-check of clean-set values (P2.3) | Hard evidence the gold is right, beyond "the program runs" |
| **A4** | Train-overlap analysis (P6.4) | Gives H1's fine-tuning pathway a direct measurement (the proposal names it but does not say how to measure it) |
| **A5** | Clean items as known non-members for membership inference (P6.3) | Calibrates the contamination detector instead of guessing a threshold |
| **A6** | Confidence intervals, paired tests, Holm correction (P1.5) | Needed for any claim about gaps of a few points; 400 clean items give ±~5.7-point resolution on a gap |
| **A7** | Train-once/evaluate-many, run reports, tracker (P1.1, P0.6, P0.7) | Engineering only: halves compute, makes every run inspectable |

Not added, on purpose: new datasets, new model families, retrieval, new training
methods. They would dilute the measurement story.

---

## 5. Compute budget (Tier 1, Kaggle)

Basis: measured 3B training 310 tok/s → ~6 h per FinQA epoch (6.7M tokens);
answering ~1 question/s (3B, answer-only); 7B ≈ 15 h/epoch and 8B ≈ 16 h/epoch
scaled by parameter count; 32B unmeasured. Evaluation per trained run covers
standard (1,147) + clean (~400) + control (~400) ≈ 1,950 questions.

| Phase | Runs | GPU-h |
|---|---|---|
| P1 pilots | throughput + smoke | ~3 |
| P3 (3B) | A5/A6/A7 × 2 seeds + A1 + A2, with dev checks and 3-split eval | ~64 |
| P4 (32B + A8-3B) | A3, A4, traces, A8 × 2 seeds | ~38 |
| P5 (7B, 8B) | must + should runs | ~124 |
| P6 | post-hoc proxy passes + membership inference | ~24 |
| **Total** | | **≈ 250** (≈ 8–9 weeks at 30 h/week) |

Levers, in order: the P1.4 speed-up (est. 1.3–2×); running two students at
once, one per T4 (only if Kaggle bills T4×2 as one GPU-hour per hour: to be
verified in P1.4); Colab's separate free T4 pool (proposal §7); then
dropping **C** and **S** runs. Rule: if spend at any gate exceeds its estimate
by >20%, stop and re-plan with you before continuing.

Statistical power: with ~400 clean and 1,147 standard items, a standard-vs-clean
gap has a standard error of ~2.9 points at 50% accuracy, so gaps of about
6 points or more are detectable at 95% confidence. This sets the 400-item
target.

---

## 6. How we keep track

1. **This file** is the single plan: phases, task IDs, gates, status. I update
   it at the end of every working session and commit it with the task ID in the
   commit message.
2. **`python scripts/roadmap.py`** (P0.7): phase progress, next tasks, GPU-hours
   used vs budget. The Kaggle notebook prints it at the start of every session.
3. **Run log** (section 8 below): one line per Kaggle session: date, `Results/`
   folder, task IDs, runs completed / failed, GPU-hours. I fill it in when you
   give me a Results folder.
4. **Decision log** (section 9 below): every decision and amendment, with the
   date and your approval. Nothing outside this plan is built without an entry.
5. **`docs/PROTOCOL.md`** (P1.9): frozen analysis protocol; changes only through
   the decision log.
6. **My memory**: once you approve, I store the phases, gates and working rules
   as linked wiki pages, so the plan survives across conversations. The repo
   file stays the authority for status.
7. **Every message** where I report work states: current phase and task ID,
   what was done, what's next, and anything I need from you.

Working rules (binding after approval):
- Work only on tasks in the current phase, or on tasks explicitly marked as
  parallel (P2 runs alongside P3–P5; no-GPU parts of P6 run alongside P5).
- A phase closes only when its gate passes, with evidence in the run log.
- New ideas, changes, or dropped tasks → an amendment proposal, then your
  approval, then a decision-log entry.
- Paid compute (P8) needs your explicit go-ahead with the cost stated.

---

## 7. Timeline (from 2026-09-22; at ~30 GPU-h/week)

| Weeks | Dates (2026–27) | GPU work | Parallel, no GPU |
|---|---|---|---|
| 1 | Sep 22–28 | P0 | — |
| 2 | Sep 29–Oct 5 | P1.4, P1.10 | P1, start P2 |
| 3–4 | Oct 6–19 | P3 | P2 |
| 5–6 | Oct 20–Nov 2 | P4, P3.7 | P2 done (G2), P6.1, P6.4 |
| 7–10 | Nov 3–30 | P5 | P9.1 outline |
| 11 | Dec 1–7 | P6.2, P6.3 | P7 starts |
| 12 | Dec 8–14 | — | P7 (G7), P8.1, P8 rented A100 (~1 day of compute) |
| 13–16 | Dec 15–Jan 11 | P8.3 | P9 draft, supervisor review, arXiv |
| — | Feb 2027 | — | Submission (ARR cycle) |

If two T4s can run two students at once, weeks 3–10 shrink to about 5 and the
whole plan fits the proposal's 12 weeks.

---

## 8. Run log

| Date | Results folder | Tasks | Outcome | GPU-h |
|---|---|---|---|---|
| 2026-09-25 | (Kaggle log only) | P0.5 | tests failed: torchao 0.10 vs peft (fixed) | 0.03 |
| 2026-09-25 | 25-9-26 2000Hrs | P0.5 | smoke failed: DataParallel over two T4s (fixed) | 0.05 |
| 2026-09-25 | 25-9-26 2030Hrs | P0.5 | smoke failed: trl bf16 cast on T4 (fixed) | 0.06 |
| 2026-09-26 | 26-9-26 1500Hrs | P0.5 | **smoke completed**: EM 0% strict (15.6% within 1%), 310 tok/s | 0.13 |


## 9. Decision log

| ID | Date | Decision | Approved |
|---|---|---|---|
| D1 | 2026-09-25 | `max_seq_len` 2048 (longer training items dropped, eval never truncated) | yes (you) |
| D2 | 2026-09-26 | Roadmap v1.0 with amendments M1–M8 and additions A1–A7 | yes (you) |

---

## 10. Publication outlook (judgment, not a guarantee)

Assuming P0–P7 and P9 are completed as planned (P8 optional but helpful):

| Outcome | Probability |
|---|---|
| arXiv preprint | ~95% |
| At least one peer-reviewed venue within ~12 months (workshop or better) | ~70–75% |
| Findings of ACL / EMNLP | ~25–35% |
| ACL / EMNLP main conference | ~8–12% |

What moves these numbers most:
- **Up:** a clean set that survives the spot-check and has the style control;
  a clear H1 result with CIs; releasing the set; the two-scale (32B + 72B)
  frontier.
- **Down:** being pre-empted in this active area (the P8.1 and P9.5 checks);
  small or null gaps without CIs to show they are real nulls; skipping the
  clean set (without it, peer review is very unlikely).
