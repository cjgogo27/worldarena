# Preference-Pair Benchmark Protocol v1 (2026-03-28)

This protocol defines the exact, reproducible, and auditable construction process for DPO training pairs.

## 1. Scope
- Target task: style transfer preference learning.
- Pair format: chosen/rejected.
- Main ratio: 1 positive : 3 negatives (1:3).

## 2. Split Before Pairing (Leakage Control)
- Split by content_id first, then construct pairs.
- Recommended split: train/val/test = 8/1/1.
- No content_id overlap across splits.
- All evaluation metrics reported on val/test only.

## 3. Condition Group Definition
A condition group is:
- (split, content_id, style_id, prompt_template_id)

All candidate images in the group must share the same condition values, except random seed.

## 4. Candidate Generation Requirement
- Recommended candidate count per group: K >= 6.
- Minimum viable candidate count: K = 4.
- Candidate seeds must be unique and logged.

## 5. Scoring and Ranking
- Primary scorer: VLM.
- Auxiliary scorer: CLIP (for logging/diagnostics only).
- Scoring determinism requirements:
  - set inference temperature to 0 (or equivalent deterministic mode)
  - fix scorer prompt template version and model version in manifest
  - retry failed scorer calls with bounded retry count; log retries
- Ranking key (deterministic):
  1) score descending
  2) seed ascending (tie-break)
  3) file_path lexicographic ascending (second tie-break)

## 6. 1:3 Pair Selection Rule
Given sorted candidates x_(1)...x_(K):
- chosen = x_(1)
- rejected_list = {x_(K), x_(K-1), x_(K-2)}
- require chosen_score > rejected_score for each rejected

## 7. Quality Filters
Filter candidates before ranking:
- unreadable/corrupted image
- duplicated image by perceptual hash (pHash threshold must be logged)
- invalid scorer output (NaN/None)

Margin rule after ranking:
- chosen_score - rejected_score >= delta_min
- recommended delta_min = 0.02 (configurable; must be reported)
- if a group has near-flat scores (max_score - min_score < 1e-6), drop group as uninformative

## 8. Fallback Rule
If valid candidates are insufficient after filtering:
- 3 candidates => fallback 1:2
- 2 candidates => fallback 1:1
- <2 candidates => drop group

All fallback and dropped groups must be reported with reasons.

## 9. Expansion to DPO Training Pairs
For one group with m negatives (m in {1,2,3}):
- expand to m training pairs: (chosen, rejected_j)

Weighting rule (critical):
- each pair weight = 1/m
- group total weight = 1.0

This avoids over-weighting groups that have more negatives.

## 10. Required Output Files
- results/pair_benchmark_build/preference_pairs.json
- results/pair_benchmark_build/pair_manifest.json
- results/pair_benchmark_build/pair_quality_report.txt
- results/pair_benchmark_build/pair_build_failures.jsonl

## 11. JSON Schema Requirements (minimum fields)
Per group record in preference_pairs.json:
- schema_version
- split
- group_id
- content_id
- style_id
- prompt_template_id
- prompt
- chosen: {image_path, score, seed}
- rejected_list: [{image_path, score, seed}, ...]
- ratio_type: one of ["1:3", "1:2", "1:1"]
- scorer: {primary, auxiliary}
- delta_min

Manifest requirements:
- total_groups
- total_pairs_expanded
- ratio distribution
- per-style coverage
- per-split coverage
- scorer model/version
- config hash
- dedup policy (hash type + threshold)
- delta_min value
- fallback counts by type (1:2, 1:1, dropped)

## 12. Hard Acceptance Gates
- valid expanded pair count >= 1000
- no illegal pair (chosen_score <= rejected_score)
- no cross-split content_id leakage
- ratio distribution reported
- fallback ratio reported
- dropped-group ratio <= 30% (if higher, stop and diagnose pipeline)
- all files have checksums in manifest

## 13. Reproducibility Requirements
- fix random seeds for generation and scoring calls when possible
- persist full command lines
- store code commit/hash or package version in manifest
- store model identifiers for VLM/CLIP scorers

## 14. Failure Handling
If VLM is unavailable:
- do not silently switch to CLIP-only
- mark run as partial/invalid for final benchmark
- emit explicit warning in pair_quality_report

If VLM and CLIP rank disagreement is extreme:
- compute disagreement rate on sampled groups
- if disagreement exceeds configured threshold, report and pause training start
- do not auto-ignore disagreement without report

## 15. Minimal Example (conceptual)
For one group with 6 candidates:
- scores: [0.81, 0.77, 0.73, 0.62, 0.55, 0.48]
- chosen: 0.81
- negatives: 0.48, 0.55, 0.62
- expanded pairs: 3
- each pair weight: 1/3
