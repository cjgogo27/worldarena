# Paper Section Draft: Benchmark + DPO as Core Contributions

This document is written as a paper-ready section draft. It can be directly adapted into Method, Benchmark, and Experimental Setup sections.

## 1. Contribution Statement

We make five concrete contributions:

1. **Benchmark Construction Contribution**: a style-transfer benchmark pipeline with explicit style coverage, fixed protocol, and reproducible data governance.
2. **Evaluation Paradigm Contribution**: we replace the original CLIP-dominant evaluation with a VLM-centered protocol that jointly evaluates style fidelity and content preservation.
3. **Preference Data Contribution (Primary)**: a preference-pair benchmark construction workflow (chosen/rejected) with strict pair validity checks, fixed positive-negative ratio protocol, and audit artifacts.
4. **Optimization Contribution**: a DPO training framework for BAGEL that learns from relative preferences rather than pointwise supervision.
5. **Engineering/Reproducibility Contribution**: a practical integration recipe covering per-sample aggregation, dtype safety, memory-aware freezing/LoRA strategy, and traceable experiment artifacts.

Together, these contributions form an end-to-end system: benchmark and evaluator define reliable supervision, preference pairs instantiate trainable signals, and DPO converts those signals into model improvements.

## 2. Benchmark Contribution

### 2.1 Benchmark Design Goal

The benchmark is designed to measure and train style transfer models under two simultaneous constraints:
- Style fidelity: generated image should reflect target style characteristics.
- Content preservation: generated image should preserve structural content from source content image.

Unlike single-metric evaluation, this benchmark enforces dual-objective consistency and supports both training-time pair construction and evaluation-time model comparison.

### 2.2 Fixed Style Benchmark (40 x 10)

We provide a fixed style benchmark with:
- 40 style categories
- 10 reference images per category
- Total of 400 benchmark style images

Package location:
- benchmark/style_benchmark_40x10/

Benchmark documentation:
- BENCHMARK_STYLE40x10_20260328.md

This fixed benchmark is used as a stable anchor to reduce protocol drift across experiments and ablation runs.

### 2.3 Preference-Pair Benchmark Construction (Primary Contribution)

Beyond fixed style references, we define a preference-pair benchmark for DPO training:
- each training unit is a pair (chosen, rejected)
- both samples are generated under matched condition groups (same prompt/condition family)
- chosen must be preferred by scorer over rejected
- **pairing ratio protocol**: each positive sample is paired with 3 negatives (`1:3`) within the same condition group

Negative selection protocol:
- rank candidates by scorer under the same condition group
- take the top-scored sample as chosen
- take the lowest 3 scored samples as rejected set for that chosen sample
- if a condition group cannot satisfy `1:3`, fallback to `1:2` or `1:1` and log this fallback in quality report

Required outputs for pair benchmark construction:
- results/pair_benchmark_build/preference_pairs.json
- results/pair_benchmark_build/pair_manifest.json
- results/pair_benchmark_build/pair_quality_report.txt
- protocol spec: PAIR_BENCHMARK_PROTOCOL_V1_20260328.md

Recommended minimum quality gates:
- valid pair count >= 1000
- no illegal pairs where chosen_score <= rejected_score
- ratio reporting required: share of `1:3` / `1:2` / `1:1` samples must be reported
- per-style coverage statistics reported

This benchmark design transforms subjective style transfer quality into auditable preference supervision.

#### 2.3.1 Exact Construction Protocol (1 Positive : 3 Negatives)

We use a deterministic group-wise pipeline.

Step 1: Build condition group
- A condition group is uniquely defined by `(content_id, style_id, prompt_template_id)`.
- For each group, generate `K` candidate images with different seeds (recommended `K >= 6`, minimum `K = 4`).

Step 2: Score all candidates
- Use VLM as primary scorer to assign score `s_i` to each candidate in the same group.
- CLIP may be logged as auxiliary score, but ranking is based on VLM by default.

Step 3: Rank and select chosen/rejected
- Sort candidates by VLM score descending:
	`x_(1), x_(2), ..., x_(K)` with `s_(1) >= s_(2) >= ... >= s_(K)`.
- Chosen (positive): `x_(1)`.
- Rejected set (negatives): `{x_(K), x_(K-1), x_(K-2)}` (lowest three scores).
- This gives one chosen with three rejected samples (`1:3`).

Step 4: Validity checks
- Enforce `s_chosen > s_rejected_j` for each negative `j`.
- Enforce score margin: `s_chosen - s_rejected_j >= delta_min` (recommended `delta_min = 0.02`; report exact value).
- Remove corrupted/invalid images before ranking.

Step 5: Fallback policy (if group has too few valid candidates)
- If valid candidates are 3: use `1:2`.
- If valid candidates are 2: use `1:1`.
- If valid candidates < 2: drop this group.
- All fallback events must be logged in `pair_quality_report.txt`.

Step 5.5: Split and leakage control (critical)
- Split by `content_id` before pair construction.
- Ensure no `content_id` overlap across train/val/test.
- Report split statistics in `pair_manifest.json`.

Step 6: Export format
- Store pair instances with explicit group id and scorer metadata.
- Recommended structure for `preference_pairs.json`:
	- `group_id`
	- `chosen: {image_path, score, seed}`
	- `rejected_list: [{image_path, score, seed}, ...]`
	- `scorer: {primary: "vlm", auxiliary: "clip"}`
	- `prompt`, `style_id`, `content_id`

Step 7: Pair expansion for DPO training
- Expand one `1:3` group record into three DPO training pairs:
	- `(chosen, rejected_1)`
	- `(chosen, rejected_2)`
	- `(chosen, rejected_3)`
- This keeps DPO input format simple while preserving multi-negative signal.
- Weighting rule: if one group has `m` negatives, each expanded pair gets weight `1/m`, so group total weight is 1. This avoids bias toward groups with more negatives.

Step 8: Deterministic tie-breaking and dedup
- For equal scorer values, tie-break by seed, then file path.
- Apply pHash-based near-duplicate filtering before final ranking.
- Record dedup threshold and removed sample count in quality report.

### 2.4 Evaluation Paradigm Shift: From CLIP-Only to VLM-Centered Protocol

An important contribution of this work is methodological: we challenge and replace the previous CLIP-only (or CLIP-dominant) evaluation style.

- **Why CLIP-only is insufficient**:
	- It tends to over-simplify style transfer quality into a single embedding similarity view.
	- It is weaker at capturing nuanced style semantics under strict content-preservation constraints.
- **What we changed**:
	- We adopt VLM as the primary evaluator for both preference pair construction and final model assessment.
	- CLIP is retained as auxiliary/efficiency baseline, not the final authority.
- **What this enables**:
	- Better alignment with human comparative judgment in style transfer.
	- Higher reliability of chosen/rejected labels used by DPO.
	- A more defensible reviewer-facing evaluation protocol.

This is not a minor metric swap; it is a benchmark protocol redesign that materially changes data quality and training signal quality.

## 3. DPO Contribution

### 3.1 Why DPO for Style Transfer

Conventional pointwise training optimizes absolute targets, while style transfer quality is inherently comparative. DPO directly optimizes relative preference between chosen and rejected samples, matching the nature of human and VLM-based judgments.

### 3.2 DPO Training Signal Definition

For each condition group c, we construct:
- preferred sample x_w (chosen)
- less preferred sample x_l (rejected)

DPO objective follows preference margin optimization:

$$
\mathcal{L}_{DPO} = -\log\sigma\left(\beta\left[(\log\pi_\theta(x_w|c)-\log\pi_\theta(x_l|c))-(\log\pi_{ref}(x_w|c)-\log\pi_{ref}(x_l|c))\right]\right)
$$

In our BAGEL integration, we emphasize per-sample aggregation in visual token losses (instead of global mixed token averaging), which is critical for variable-length visual token regimes.

### 3.3 Engineering Contribution in BAGEL Context

Our contribution is not only the objective function but also practical integration under real constraints:
- policy/ref dual-branch training workflow
- dtype consistency controls (bf16/float mismatch mitigation)
- freeze/LoRA configurable strategy for memory-constrained multi-GPU training
- compatibility with existing BAGEL training skeleton

Reference code paths in package:
- code/train/qwen2_dpo_reference.py
- code/repos/bagel-main/train/pretrain_unified_navit.py
- code/repos/bagel-main/modeling/bagel/bagel.py

Target entry script (to be finalized in server workflow):
- code/train/dpo_training.py

## 4. Unified Experimental Protocol (for Reviewers)

### 4.1 Mandatory Pipeline Order

1. Build and validate fixed style benchmark (40x10).
2. Build and validate preference-pair benchmark.
3. Run DPO training on preference pairs.
4. Evaluate with CLIP + VLM protocols.
5. Report both global metrics and per-style breakdown.

This order is enforced in deployment prompt and quality gates, preventing premature training on unvalidated data.

### 4.2 Scorer Layer and Model Layer Separation

We explicitly separate:
- Scorer baselines: CLIP-only, VLM-only, Hybrid (CLIP->VLM)
- Model baselines: BAGEL Base, BAGEL+LoRA, BAGEL+DPO, BAGEL+LoRA+DPO
- Ablations: aggregation strategy, beta, freezing policy, LoRA rank, inference hyperparameters

In paper writing, emphasize that "scorer is a first-class design choice" rather than an implementation detail; this directly supports the evaluation-paradigm contribution.

Baseline specification is documented in:
- BASELINES_20260328.md

### 4.3 Reproducibility and Auditability

We treat data and experiment artifacts as first-class outputs:
- benchmark manifests
- pair quality reports
- deterministic command traces
- checkpoint and evaluation logs

This makes our benchmark and DPO results independently inspectable and reproducible.

## 5. Evidence Map (What Exists in Current Package)

- Baseline protocol: BASELINES_20260328.md
- Missing-file audit: MISSING_FILES_CHECKLIST_20260328.md
- Project handover: PROJECT_HANDOVER_20260328.md
- Benchmark description: BENCHMARK_STYLE40x10_20260328.md
- Style benchmark data: benchmark/style_benchmark_40x10/
- Claude execution protocol (with pre-training benchmark gates): CLAUDE_SERVER_PROMPT_20260328.md

## 6. Suggested Paper Wording (Concise Version)

### 6.1 Contribution Paragraph (can be used in Introduction)

We contribute a full preference-learning stack for style transfer: (i) a reproducible benchmark pipeline with fixed style coverage, (ii) an evaluation paradigm shift from CLIP-dominant scoring to VLM-centered assessment, (iii) a validated preference-pair benchmark construction workflow, and (iv) a practical DPO integration for BAGEL under real multi-GPU constraints. This design improves supervision reliability and produces auditable, reproducible training/evaluation artifacts.

### 6.2 Method Paragraph (can be used in Method)

Given matched condition groups, we construct chosen/rejected pairs using scorer-guided ranking and optimize a DPO objective against a reference policy. Unlike token-mixed averaging, we adopt per-sample aggregation to preserve fairness across variable-length visual token sequences. This design is essential for stable preference optimization in multimodal style transfer.

### 6.3 Benchmark Paragraph (can be used in Benchmark Section)

Our benchmark has two layers: (i) a fixed 40x10 style reference set for protocol stability; and (ii) a preference-pair benchmark for DPO training, with strict validity checks (chosen_score > rejected_score, per-style coverage, and quality manifests). Crucially, we redesign the evaluation protocol by moving from CLIP-dominant scoring to VLM-centered assessment, making benchmark labels and comparisons more faithful to style-transfer objectives.

### 6.4 Evaluation-Shift Paragraph (can be used in Method/Experiment)

Previous pipelines commonly relied on CLIP as the primary evaluator. We show this is insufficient for high-fidelity style transfer where style semantics and content preservation must be judged jointly. Therefore, we elevate VLM to the primary role in both pair construction and final evaluation, while retaining CLIP as an auxiliary baseline for efficiency. This protocol change is a core contribution because it reshapes the training data distribution and the reliability of DPO preference signals.

## 7. Reviewer-Facing Checklist

Before submission, ensure the paper includes:
- exact benchmark construction rules
- pair validity constraints and statistics
- scorer/model/ablation baseline taxonomy
- reproducibility artifacts (manifest/report paths)
- failure case analysis and limitations

## 8. Notes on Claims

To keep claims rigorous:
- claim what is verified by artifacts and logs
- separate "implemented" vs "planned" components clearly
- avoid over-claiming absolute SOTA unless fully benchmarked under public protocol

---

If needed, this draft can be split into three paper subsections:
- Section A: Benchmark Construction
- Section B: DPO Integration in BAGEL
- Section C: Reproducible Evaluation Protocol
