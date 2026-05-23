# GeoSkill Project Memory

## Current Status (2026-03-11 01:25 UTC)

## Quick Session Learnings (2026-03-20)

- Added three standalone point-cloud diffusion demos under `/data/alice/cjtest/demos`:
  - `demo1_1d_particle_flow`
  - `demo2_2d_shape_morph`
  - `demo3_coordinate_digit`
- Each demo is self-contained with `run.py`, outputs static PNG figures and `metrics.json`.
- For sharing diffs in non-git parent directories, initialized local git only inside the target folder (`/data/alice/cjtest/demos`) to enable `bunx critique --web`.
- Oracle subagent occasionally returns empty assistant payload in session read despite completion logs; do not block core delivery on that path.

### API BLOCKER (still active as of 01:25 UTC)
`nowcoding.ai/v1` returning HTTP 400/500 for ALL models since ~Mar 11 00:46 UTC.
- **Action needed**: Recharge account at nowcoding.ai or provide new API key
- **Auto-run**: tmux session `geoskill-v3v4` is running `scripts/wait_api_and_run_v3_v4.sh`
  - Polls API every 60s; auto-launches v3 smoke → v3 full → v4 smoke → v4 full → recompute metrics
  - Monitor log: `experiments/full_100/v3v4_monitor.log`

### v4 Implementation: COMPLETE (2026-03-11 01:25 UTC)
`skill_conditioned_v4_predict()` added to `src/baselines.py`.
Wired into `scripts/run_experiment.py` (all_methods dict) and `scripts/recompute_metrics_v2.py` (METHODS list).
Auto-run script: `scripts/wait_api_and_run_v3_v4.sh`

### v3 Implementation Status: COMPLETE, NEEDS RUN
All code is implemented and ready. Only the API outage prevents the experiment from running.
The auto-run script handles this.

## v4 Design (Implemented in src/baselines.py, 2026-03-11)

`skill_conditioned_v4_predict()` — 7-stage self-consistency + hierarchical + self-critique pipeline:
1. **Pass A (GeoCoT, temp=0.2)**: Standard 5-step structured reasoning (proven v2/v3 Stage 1 prompt)
2. **Pass B (Hierarchical, temp=0.3)**: PIGEON-style forced continent→country→city hierarchy — prevents continent errors
3. **Pass C (Adversarial, temp=0.4)**: "List top 3 candidates, find the 1 clue that rules each out" — eliminates confusable countries
4. **Local detail fingerprint (GRE-style, temp=0.1)**: pole/road/sign JSON (same as v3 Stage 2)
5. **Majority vote**: 2+/3 passes agree on country → voted_country/region; if split 3-way, use local detail → fall back to Pass A
6. **Coordinate averaging**: Average lat/lng from agreeing passes as coordinate baseline
7. **Text-only self-critique synthesis (temp=0.1)**: Given Pass A+B+C summaries + local fingerprint + 12 retrieved skills → critique + final answer

Key design choices:
- 4 image calls (A, B, C, local) + 1 text-only call = 5 total VLM calls per sample (vs 3 for v3, 2 for v2)
- Top_k=12 skills retrieved (vs 10 for v3, 8 for v2)  
- Coordinate baseline from averaging agreeing passes reduces outlier error
- Pass B specifically targets continent-level errors (Panama→Thailand type mistakes from v2)
- Pass C specifically targets country-confusion errors (Turkey→South Africa type from v2)
- Final synthesis at temperature=0.1 (lowest) for stability

## Results (as of 2026-03-10, v2 is latest complete)

### skill_conditioned_v2 vs All Baselines

| Method | Country | Region | Dist mean | Dist median | Valid% |
|--------|---------|--------|-----------|-------------|--------|
| direct_vlm | 0.50 | 0.81 | 3,039 km | 706 km | 94% |
| cot_vlm | 0.42 | 0.71 | 5,946 km | 1,084 km | 79% |
| georeasoner | 0.49 | 0.86 | 2,131 km | 671 km | 100% |
| geocot | **0.54** | **0.88** | 2,606 km | 689 km | 98% |
| gre_multistage | 0.49 | **0.90** | **1,925 km** | 698 km | 100% |
| img2loc_rag | 0.47 | 0.85 | 2,364 km | 748 km | 100% |
| skill_conditioned v1 | 0.44 | 0.83 | 2,541 km | 874 km | 100% |
| **skill_conditioned_v2** | **0.54** | 0.87 | 2,253 km | **628 km** | **100%** |

### Remaining gaps for v3 to close:
- Country accuracy: > 0.54 (tied, need strict best)
- Region accuracy: > 0.90 (GRE leads at 0.90)
- Mean distance: < 1,925 km (GRE leads)

## v3 Design (Implemented in src/baselines.py)

5-stage pipeline `skill_conditioned_v3_predict()`:
1. **GeoCoT** (image) → candidate country + region (same as v2)
2. **GRE local detail** (image) → pole/road/sign fingerprint JSON + `implied_country` + `implied_region`
3. **Cross-validate**: if stage2 region ≠ stage1 region → use stage2 (overrides geocot to fix continent errors like Panama→Thailand)
4. **Expanded skill retrieval**: `top_k=10`, synth_query = stage1_out[:500] + local_info JSON[:300] + active_country + active_region. Uses expanded library (~500+ skills from both expert + candidate chains)
5. **Text-only synthesis** (NO image resubmit — avoids 400 token-limit errors): stage1 summary + local detail + 10 skills → final JSON

### Key v3 code decisions:
- Final call uses `image_path=None` to avoid context length 400 errors
- Skills truncated to 120 chars each in final prompt
- Stage1 truncated to 400 chars in final prompt
- Local detail JSON truncated to 350 chars

## Expanded Skill Library (v3)

`build_skill_library()` in `run_experiment.py` now also parses `candidate_reasoning_chain_gpt4_{1-5}.txt` per game via `parse_candidate_chain()` in `skill_parser.py`.

Format: plain observation lines (no Round/Conclusion headers). Last line with a known country → region_hint. Lines < 15 chars skipped. Expect ~500+ skills (vs 183 from expert chains only).

## Key Technical Learnings

### skill_conditioned_v2 Design (what made it work)
1. **GeoCoT first pass** — structured 5-step reasoning (hemisphere→continent→country→region→coords) produces reliable region anchor for retrieval
2. **Region-targeted retrieval** — `deduplicate_region=False`, filter by predicted region, sort by skill length descending (composed > atomic)
3. **Strong prior framing** — "treat as strong priors from expert players" not "ignore if contradicts"
4. **Rich query** — GeoCoT full reasoning text + candidate country/region as retrieval query (not bare scene description)
5. **Fallback** — if < 3 region-matching skills, fall back to global top-k

### Why v1 failed (44%)
- Scene description query caused cross-region skill drift (e.g., Italy image retrieves Vietnam skills)
- `deduplicate_region=True` prevented getting multiple skills for the correct region
- Permissive "IGNORE if contradicts" caused model to discard skills entirely
- Atomic skills added noise vs composed multi-cue skills

### Root cause of v2 remaining gaps
- **Continent-level errors**: game `d8lnc9Ex` (gt=Panama), v2 predicts Thailand (17,079 km); game `z2mhsiTu` (gt=Panama), v2 predicts unknown (14,422 km); game `N4EN1CWs` (gt=Turkey), v2 predicts South Africa (7,468 km). GRE's local detail pass prevents these.
- **v2 and GeoCoT predict identically** — stage 3 of v2 rarely overrides stage 1; need cross-validation signal

### Ablation results (all complete)
| Variant | Country | Delta vs v2 |
|---------|---------|-------------|
| Full v2 | 0.54 | --- |
| No skill | 0.32 | -0.22 |
| Random skills | 0.40 | -0.14 |
| Shuffled order | 0.39 | -0.15 |
| Atomic only | 0.42 | -0.12 |
| Composed only | 0.47 | -0.07 |

### Code Patterns
- `SkillLibrary(embedding_model_name: str)` — string, not dict
- `skill_library.retrieve(query_text, top_k, alpha, min_score, deduplicate_region)`
- `parse_expert_chain(text, source_game_id)` — positional, not keyword
- `parse_candidate_chain(text, source_game_id, round_num)` — NEW in v3
- `VLMClient.extract_json()` — static method
- Images: resize to max 1024px JPEG, RGBA→RGB before VLM call
- `COUNTRY_TO_REGION[country_code]` — region lookup; region accuracy always ≥ country accuracy
- **CRITICAL**: Do NOT pass image to final synthesis stage (causes 400 context-length errors)

### Environment
- Server: `/data/alice/cjtest/geoskill/`
- Conda env: `openclaw` (Python 3.13.11)
- VLM: `nowcoding.ai/v1`, model `claude-sonnet-4-5`
- SentenceTransformer must use `device="cpu"` (GPUs saturated)
- Compile paper: `cd paper && PATH="$HOME/.local/bin:$PATH" tectonic neurips_2026.tex`

### Feishu Reporting
- User: `ou_7246fdeaef7196587c67994fa7029b3c`
- Group: `oc_9b92545c33c159922c678d77715e2cc2`
- App ID: `cli_a92401e4eae11bc9`
- App Secret: `hwF6AObg3hY72OC6aie7Nbvo5yrT7008`

## Paper Status
- `paper/neurips_2026.tex` — fully updated with v2 results (needs v3 update after run)
- `paper/neurips_2026.pdf` — 83.16 KiB, compiles clean
- Table 1: skill_conditioned_v2 row added, bold/underline updated
- Table 2 (ablation): delta values updated vs v2 baseline
- Table 3 (cross-region): v2 row added (europe=0.43, asia=0.61, n_am=0.55, s_am=0.64, africa=0.60, oceania=1.00)
- Abstract, conclusion, method section: all updated to reflect v2 design


### skill_conditioned_v2 vs All Baselines

| Method | Country | Region | Dist mean | Dist median | Valid% |
|--------|---------|--------|-----------|-------------|--------|
| direct_vlm | 0.50 | 0.81 | 3,039 km | 706 km | 94% |
| cot_vlm | 0.42 | 0.71 | 5,946 km | 1,084 km | 79% |
| georeasoner | 0.49 | 0.86 | 2,131 km | 671 km | 100% |
| geocot | **0.54** | **0.88** | 2,606 km | 689 km | 98% |
| gre_multistage | 0.49 | **0.90** | **1,925 km** | 698 km | 100% |
| img2loc_rag | 0.47 | 0.85 | 2,364 km | 748 km | 100% |
| skill_conditioned v1 | 0.44 | 0.83 | 2,541 km | 874 km | 100% |
| **skill_conditioned_v2** | **0.54** | 0.87 | 2,253 km | **628 km** | **100%** |

### Our method leads on:
- Median distance: 628 km (best of all, beats GeoCoT 689 km)
- Valid coordinate rate: 100% (GeoCoT only 98%)
- Country accuracy: 0.54 (tied best with GeoCoT)

### GRE Multi-stage still leads on:
- Mean distance: 1,925 km
- Region accuracy: 0.90

## Key Technical Learnings

### skill_conditioned_v2 Design (what made it work)
1. **GeoCoT first pass** — structured 5-step reasoning (hemisphere→continent→country→region→coords) produces reliable region anchor for retrieval
2. **Region-targeted retrieval** — `deduplicate_region=False`, filter by predicted region, sort by skill length descending (composed > atomic)
3. **Strong prior framing** — "treat as strong priors from expert players" not "ignore if contradicts"
4. **Rich query** — GeoCoT full reasoning text + candidate country/region as retrieval query (not bare scene description)
5. **Fallback** — if < 3 region-matching skills, fall back to global top-k

### Why v1 failed (44%)
- Scene description query caused cross-region skill drift (e.g., Italy image retrieves Vietnam skills)
- `deduplicate_region=True` prevented getting multiple skills for the correct region
- Permissive "IGNORE if contradicts" caused model to discard skills entirely
- Atomic skills added noise vs composed multi-cue skills

### Ablation results (all complete)
| Variant | Country | Delta vs v2 |
|---------|---------|-------------|
| Full v2 | 0.54 | --- |
| No skill | 0.32 | -0.22 |
| Random skills | 0.40 | -0.14 |
| Shuffled order | 0.39 | -0.15 |
| Atomic only | 0.42 | -0.12 |
| Composed only | 0.47 | -0.07 |

### Code Patterns
- `SkillLibrary(embedding_model_name: str)` — string, not dict
- `skill_library.retrieve(query_text, top_k, alpha, min_score, deduplicate_region)`
- `parse_expert_chain(text, source_game_id)` — positional, not keyword
- `VLMClient.extract_json()` — static method
- Images: resize to max 1024px JPEG, RGBA→RGB before VLM call
- `COUNTRY_TO_REGION[country_code]` — region lookup; region accuracy always ≥ country accuracy

### Environment
- Server: `/data/alice/cjtest/geoskill/`
- Conda env: `openclaw` (Python 3.13.11)
- VLM: `nowcoding.ai/v1`, model `claude-sonnet-4-5`
- SentenceTransformer must use `device="cpu"` (GPUs saturated)
- Compile paper: `cd paper && PATH="$HOME/.local/bin:$PATH" tectonic neurips_2026.tex`

### Feishu Reporting
- User: `ou_7246fdeaef7196587c67994fa7029b3c`
- Group: `oc_9b92545c33c159922c678d77715e2cc2`
- App ID: `cli_a92401e4eae11bc9`
- App Secret: `hwF6AObg3hY72OC6aie7Nbvo5yrT7008`

## Paper Status
- `paper/neurips_2026.tex` — fully updated with v2 results
- `paper/neurips_2026.pdf` — 83.16 KiB, compiles clean
- Table 1: skill_conditioned_v2 row added, bold/underline updated
- Table 2 (ablation): delta values updated vs v2 baseline
- Table 3 (cross-region): v2 row added (europe=0.43, asia=0.61, n_am=0.55, s_am=0.64, africa=0.60, oceania=1.00)
- Abstract, conclusion, method section: all updated to reflect v2 design


## Session Note (2026-03-25 05:12 UTC)

- Must keep evaluator/paper/scripts in lockstep when metric names change.
- Updated evaluator schema now uses: `continent_accuracy`, `expert_chain_token_f1`, `heuristic_hallucination_rate`, `distance_error_km_median`, `distance_error_km_mean_valid_only`, `distance_error_km_mean_penalized`, and `Acc@{1,25,150,750,2500}km`.
- Empty input handling in evaluator is now strict (`raise ValueError`), no `n=1` fallback.
- `paper/neurips_2026.tex` baselines section and main table were rewritten to the requested 4-baseline protocol (GeoReasoner/GeoVista/GeoComp(SAFA)) with placeholder rows for baselines not yet executed locally.
- compileall caught a pre-existing syntax issue in `scripts/generate_figures.py` f-string; fixed with `scene_preview` intermediate variable.
- This repo was initialized locally with `.git` but has no commits yet; all files appear untracked in current local git state.

## Session Note (2026-03-26 07:40 UTC)

- Dataset inventory re-verified: `configs/full.yaml` has 100 game_ids, `configs/pilot.yaml` has 20, and `data/georc` has 100 game folders.
- Candidate trajectory coverage is incomplete: only 20 games have full `candidate_reasoning_chain_gpt4_{1..5}.txt`; total candidate files = 100, meaning 80 games are missing candidate chains.
- External baseline folders (`GeoReasoner`, `GeoVista`, `GeoComp`, `SAFA`) exist, but project-local experiment artifacts for those baselines are absent (`run_audit.json` shows `local_results_found=false`); current state is consistent with cloned code, not completed local baseline runs.
- Main immediate runtime blocker changed from timeouts to auth: fresh smoke reruns fail with HTTP 401 invalid token (`无效的令牌`) across methods.
- Portability + pipeline robustness patches applied:
  - Removed hardcoded `/data/alice/cjtest/geoskill` paths in scripts; switched to repo-relative root derivation.
  - `run_experiment.py` and `run_ablation.py` now support `VLM_API_KEY` env override.
  - v3/v4 final synthesis path now retries with concise image-backed fallback when text-only final query fails.
  - `run_audit.json` external baseline scan now includes script/artefact indicators.
