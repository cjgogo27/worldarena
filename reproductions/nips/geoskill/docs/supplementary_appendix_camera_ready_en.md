# Appendix: Reproducibility, Training-Free Skill Evolution, and Failure-Case Board

This appendix is generated from repository artifacts under `experiments/`, `configs/`, and `docs/`, and is intended to be submission-ready with minimal manual edits.

## A. Reproducibility Snapshot

- Workspace root: `/data/alice/cjtest/NIPS/geoskill`
- Primary experiment artifacts: `experiments/full_100_provider_a`, `experiments/review_evolution_georc50_provider_a`, `experiments/ablation_geoskill_skill_graph_georc50_rollout3`.
- Audit files confirming loaded sample counts and method-level success/error rates are available at:
  - `experiments/full_100_provider_a/run_audit.json`
  - `experiments/review_evolution_georc50_provider_a/run_audit.json`
  - `experiments/ablation_geoskill_skill_graph_georc50_rollout3/run_audit.json`

### A.1 Dataset Description and Analysis

GeoSkill uses a unified sample protocol across datasets, so all methods and evaluators consume the same record structure.

1. **GeoRC (directory-based loading)**
  - Each sample is resolved from a game folder containing image files and round metadata.
  - Canonical sample ID format is `georc::<game_id>::round<r>`.

2. **EarthWhere and Im2GPS3k (manifest-based loading)**
  - Each sample is loaded from a JSONL manifest.
  - Required fields are image path and coordinates (`lat/lng` or `latitude/longitude`), while country labels are normalized to ISO-2 when possible.

3. **Unified normalized fields for evaluation and audit**
  - `sample_id`, `dataset_name`, `dataset_version`, `image_path`
  - `ground_truth_country`, `ground_truth_lat`, `ground_truth_lng`
  - optional `expert_chain`

The multi-dataset protocol and manifest requirements are specified in `configs/full_multidataset.yaml`, and the sample normalization/loading logic is implemented in `scripts/run_experiment.py`.

#### A.1.1 Sample Coverage Used in This Appendix

| Experiment | Expected Samples | Loaded Samples | Method Errors | Error Rate |
|---|---:|---:|---:|---:|
| full_100_provider_a | 50 | 50 | 0 (direct), 0 (graph) | 0.0% |
| review_evolution_georc50_provider_a | 50 | 50 | 6 | 12.0% |
| ablation_geoskill_skill_graph_georc50_rollout3 | 50 | 50 | 50 | 100.0% |

### A.2 Detailed Description of Baselines

All baselines are executed through the same experiment runner and parsed into a unified prediction schema before evaluation. This avoids metric drift caused by method-specific output formats.

| Baseline | Core Pipeline | Output Normalization |
|---|---|---|
| `direct_vlm` | single-pass geolocation prompt on image | parsed into unified JSON prediction fields |
| `cot_vlm` | single-pass with structured chain-of-thought cues | same parser and evaluator as all methods |
| `georeasoner` | two-stage reasoning (`region/top3` -> refined location) | stage outputs are preserved as auxiliary fields |
| `geocot` | five-step GeoGuessr-style structured reasoning | standardized country/region/lat/lng/confidence |
| `gre_multistage` | scene pass + local-detail pass + synthesis pass | intermediate analyses stored for auditability |
| `img2loc_rag` | scene description + retrieval + grounded final prediction | retrieved references included in output record |

In addition, external baseline adapters are supported via command wrappers (`GeoComp`, `GeoReasoner`, `GeoVista`, `SAFA`, `EP-BEV`, `Sample4Geo`) under a unified runtime interface in the config.

### A.3 Prompt Examples (Offline and Online)

To clarify reproducibility, we provide representative prompt templates used in the implementation.

#### A.3.1 Offline Prompt (Expert-to-Skill Extraction During Evolution)

```text
System:
You produce strict JSON only.

User (template excerpt):
You are a geolocation skill optimizer.
Given failed geolocation cases, extract reusable skills that reduce these failure patterns.
Return ONLY JSON array. Each element:
{"skill_text": str, "region_hint": str, "confidence": float, "visual_cues": [str]}
Constraints:
- skill_text should be concise and actionable
- include both atomic and composed skills
- region_hint one of: europe, asia, north_america, south_america, africa, oceania, unknown
- confidence in [0,1]
- prioritize robust and repeatedly-supported cues
- no markdown
Failed cases: <packed_failure_records_json>
```

This template is implemented in `src/skill_optimizer.py` (`synthesize_failure_skills`).

#### A.3.2 Online Prompt (Skill-Conditioned Inference)

```text
System (template excerpt):
You are an expert geolocation analyst. You analyze street view images to determine
the precise location. Use visual evidence: road markings, signs, vegetation,
architecture, utility poles, driving side, license plates, terrain, climate indicators.
Always provide your best guess even when uncertain.

User Stage-1 (template excerpt, GeoCoT-style):
STEP 1 - HEMISPHERE & CLIMATE ZONE
STEP 2 - CONTINENT NARROWING
STEP 3 - COUNTRY IDENTIFICATION
STEP 4 - REGION WITHIN COUNTRY
STEP 5 - FINAL LOCATION TEXT

Output constraint:
You MUST respond with ONLY a valid JSON object with keys:
country, country_code, region, city, province_or_state, address,
confidence, reasoning, evidence
```

The Stage-1 prompt, local-detail prompt, and strict JSON schema enforcement are implemented in `src/baselines.py` (`skill_conditioned_v3_predict`, shared JSON schema instruction, and geolocation system prompt).

## B. Skill Schema and Retrieval

The skill representation is a structured dataclass with six fields: `skill_text`, `region_hint`, `confidence`, `visual_cues`, `source_game_id`, and `source_round`. Retrieval uses a hybrid ranker:

\[\text{score}=\alpha\cdot\text{semantic\_norm} + (1-\alpha)\cdot\text{bm25\_norm}.\]

The online pipeline enforces a strict JSON output schema for all methods, enabling a unified evaluator and export path to GeoRC.

### B.1 Canonical Skill Record Format

To make appendix examples auditable and reusable, we keep the public skill record in the following canonical format:

```json
{
  "skill_text": "<natural-language heuristic>",
  "region_hint": "<europe|asia|north_america|south_america|africa|oceania|unknown>",
  "confidence": <float in [0,1]>,
  "visual_cues": ["cue1", "cue2", "..."],
  "source_game_id": "<origin sample or fusion id>",
  "source_round": <int>
}
```

### B.2 Representative Skill Examples Included in Appendix

From the curated Top30 skill set, we include three representative examples in the appendix body to cover composed reasoning, anti-overfitting constraints, and cross-continent ambiguity control.

| Rank in Top30 | Region Hint | Confidence | Source Kind | Representative Skill Text |
|---:|---|---:|---|---|
| 1 | asia | 0.95 | fused | Composed skill: if cue `cyrillic text` co-occurs with Soviet-style roadside design, prioritize Central Asia but avoid overcommitting to Kazakhstan without country-specific signage/plates/mountain context. |
| 7 | unknown | 0.97 | recovered | Downweight hemisphere/climate/generic utility poles; use unique text, traffic side, plate style, and sign design for country decisions. |
| 11 | south_america | 0.94 | recovered | Dry tropical scrub and red dirt alone are insufficient to switch Brazil to East Africa; verify with utility poles, signage, and vehicle conventions. |

These representative examples are also provided in machine-readable form at `docs/skill_examples_appendix_selected.jsonl` and readable table form at `docs/skill_examples_appendix_selected.md`.

### B.3 Remaining Skills as Upload Package

We provide two upload packages:

- `docs/skill_examples_appendix_selected_bundle.tar.gz` (the 3 representative skills shown in the appendix)
- `docs/skill_examples_top30_bundle.tar.gz` (full Top30 for audit and supplementary release)

Together, these packages include JSONL/CSV/Markdown exports and source `skill_updates` provenance files for external audit.

## C. Training-Free Skill Evolution Protocol

A sample is marked as failed if any of the following holds: runtime error, country mismatch, invalid coordinates, or distance error above `failure_distance_km`. Failed cases are used to synthesize new skills, optionally fuse them, and re-run failed records over rollout rounds.

### C.1 Artifact-Level Evolution Counts

| Experiment | Recovered Skills (jsonl lines) | Fused Skills (jsonl lines) | Graph Latest |
|---|---:|---:|---|
| full_100_provider_a | 12 | 3 | experiments/full_100_provider_a/skill_updates/graph_geoskill_skill_graph_latest.json |
| review_evolution_georc50_provider_a | 36 | 0 | N/A |
| ablation_geoskill_skill_graph_georc50_rollout3 | 31 | 0 | experiments/ablation_geoskill_skill_graph_georc50_rollout3/skill_updates/graph_geoskill_skill_graph_latest.json |

For `full_100_provider_a`, the latest graph snapshot reports `edge_count=80`, `num_records=5`, and `num_failed_records=3`.

For `ablation_geoskill_skill_graph_georc50_rollout3`, the latest graph snapshot reports `edge_count=0`, `num_records=50`, and `num_failed_records=50`.

## D. GeoRC Faithfulness Evaluation Protocol

Predictions are exported to GeoRC challenge-format files and then evaluated with one of three scoring modes: `key_points`, `bipartite`, or `vlm_judge`. This path is implemented in `scripts/export_predictions_to_georc.py` and `external_baselines/GeoRC/score.py`.

## E. Quantitative Results from Existing Artifacts

### E.1 Main Comparison on full_100_provider_a

| Method | Country Acc | Continent Acc | Median Dist (km) | Valid Coordinate Rate | Acc@200km | Acc@750km | Acc@2000km |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_vlm | 0.560 | 0.880 | 374.2 | 100.0% | 0.340 | 0.680 | 0.820 |
| geoskill_skill_graph | 0.580 | 0.900 | 567.9 | 100.0% | 0.300 | 0.600 | 0.800 |

### E.2 Review Evolution Run (skill_conditioned_v3, georc50)

| Metric | Value |
|---|---:|
| country_accuracy | 0.660 |
| continent_accuracy | 0.837 |
| valid_coordinate_rate | 88.0% |
| recovered_failures | 11 |
| generated_skill_count | 36 |
| rollout_rounds | 3 |
| num_errors | 6 |

### E.3 Ablation Snapshot (geovista skill-graph on georc50)

| Setting | num_errors | generated_skill_count | rollout_rounds | valid_coordinate_rate |
|---|---:|---:|---:|---:|
| w/o skill (`ablation_geoskill_skill_graph_georc50_woskill`) | 50 | 0 | 0 | 0.0% |
| rollout=1 (`ablation_geoskill_skill_graph_georc50_rollout1`) | 50 | 9 | 1 | 0.0% |
| rollout=3 (`ablation_geoskill_skill_graph_georc50_rollout3`) | 50 | 31 | 3 | 0.0% |

## F. Failure-Case Figure Board (Auto-generated)

For figure selection, we provide three machine-generated lists:

1. Full failure inventory across experiments: `docs/failure_case_figure_checklist.csv` and `docs/failure_case_figure_checklist.md`
2. Core appendix subset (two experiments): `docs/failure_case_figure_checklist_core.csv` and `docs/failure_case_figure_checklist_core.md`
3. Tier labels (`Tier-1`/`Tier-2`) are included in the core subset for quick figure curation.

The core subset contains sample ID, error type, image path, and corresponding experiment-level `skill_updates`/`graph_*_latest.json` files.

## G. Known Consistency Risks and Reporting Notes

1. **Acc@k threshold drift**: evaluator runtime currently uses `[10, 25, 200, 750, 2000]`, while docs/util defaults still mention `[1, 10, 25, 100, 200, 750, 2500]`. The camera-ready version should explicitly state which threshold set is used for reported numbers.
2. **v4 execution path**: current implementation routes `skill_conditioned_v4_predict` through a stabilized v3 path (`stabilized_via_v3`). If the paper claims full multi-pass v4 logic, this should be aligned with code or documented as a fallback mode.

## H. Author-Provided Items Still Needed

- Multi-seed variance/significance protocol (number of seeds, confidence intervals, statistical tests).
- Token-level and billing-level API cost accounting.
- Final ethics and misuse statement for camera-ready submission.

---
This appendix draft was generated directly from repository artifacts on 2026-04-09.
