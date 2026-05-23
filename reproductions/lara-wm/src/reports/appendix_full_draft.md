# LaRA-WM Appendix Draft (Project-Internal Version)

> **Scope note.** This appendix draft is assembled **only** from files under `/data/alice/cjtest/lara-wm` and is intended to be a project-internal, paper-facing consolidation of all currently available appendix material. Wherever the repository does not yet contain enough information for a paper-complete appendix, the missing item is explicitly marked as `TODO` or `Not yet available in repo`.

> **Status note.** The repository currently contains a reliable offline evaluation pipeline and partial native RoboTwin rollout infrastructure. The offline benchmark numbers are usable as measured results. The success-rate appendix sections below define the intended protocol and summarize the current state of implementation, but they should not be presented as final quantitative results until the native execution side is stabilized.

---

## A. Appendix Roadmap

This appendix is organized to cover the minimum material expected by a NeurIPS submission for a robotics / ML systems paper:

1. Method details and architecture
2. Training objectives and implementation details
3. Test-time refinement protocol
4. Benchmark, data, tasks, and success criteria
5. Baseline details and fairness notes
6. Hyperparameters and training setup
7. Additional quantitative results and result files
8. Failure analysis and qualitative evidence
9. Compute, reproducibility, and release notes
10. Limitations, safety, and asset/licensing notes

---

## B. Repository-Internal Canonical Artifact Locations

### B.1 Main draft and report artifacts

- `src/reports/neurips_draft_sections.tex` — current draft with abstract, method, and experiments sections.
- `src/reports/report.md` — current project-internal results summary.
- `src/reports/figure_snippets.tex` — current qualitative figure snippets.

### B.2 Main LaTeX tables

- `src/reports/table1_main_comparison.tex`
- `src/reports/table2_ablation.tex`
- `src/reports/table3_expanded_offline_comparison.tex`
- `src/reports/table4_zero_shot_vla_probe.tex`
- `src/reports/table5_success_metric_protocol.tex`
- `src/reports/table6_success_smoke.tex`
- `src/reports/table7_success_partial_progress.tex`
- `src/reports/table_diagnostic.tex`
- `src/reports/table_task_summary.tex`

### B.3 Main measured result files

- `experiments/results/real_training_results.json`
- `experiments/results/real_training_results.md`
- `experiments/results/real_training_results_per_task.md`
- `experiments/results/act_original_results.json`
- `experiments/results/diffusion_policy_results.json`
- `experiments/results/openvla_eval_1ep.json`
- `experiments/results/openvla_eval_3ep.json`
- `experiments/results/univla_direct_1ep.json`
- `experiments/results/univla_direct_3ep.json`
- `experiments/results/univla_latent_decoder_2tr1val_per_task_3ep_test.json`
- `experiments/results/univla_latent_decoder_2tr1val_per_task_3ep_train.json`

### B.4 Rollout / native success artifacts

- `experiments/rollout_success/`
- `experiments/rollout_success_repaired/`
- `experiments/rollout_success_repaired_v2/`
- `experiments/rollout_success_seen/`
- `experiments/rollout_success_chunked/`
- `experiments/rollout_success_dp/`
- `experiments/rollout_success_serial/`

### B.5 Rollout checkpoint artifacts

- `experiments/rollout_ckpts/`
- `experiments/rollout_ckpts_smoke/`
- `experiments/rollout_ckpts_chunked_smoke/`
- `experiments/rollout_ckpts_grab_roller_strong/`
- `experiments/diffusion_policy_robottwin/`
- `experiments/diffusion_policy_robottwin_smoke/`

---

## C. Full Method Details

### C.1 High-level model decomposition

The codebase currently implements LaRA-WM as a combination of:

1. **Latent action encoder** (VAE-like)
2. **Latent-space world model**
3. **Action decoder**
4. **Latent refinement module**
5. **Backbone adapter / feature extractor**

Primary source files:

- `src/models/latent_encoder.py`
- `src/models/world_model.py`
- `src/models/action_decoder.py`
- `src/models/latent_refinement.py`
- `src/backbone/adapter.py`
- `src/backbone/config.py`

### C.2 Architecture details from config files

From `configs/model.yaml`:

#### Latent encoder

- `action_dim: 7`
- `latent_dim: 32`
- `feature_dim: 1536`
- `hidden_dim: 256`
- `num_layers: 2`
- `dropout: 0.1`
- `kl_weight: 1.0`
- `collapse_variance_threshold: 0.001`
- `collapse_kl_threshold: 0.001`

#### World model

- `latent_dim: 1536`
- `action_dim: 1536`
- `hidden_dim: 512`
- `num_layers: 2`
- `dropout: 0.1`
- `architecture: "gru"`
- `num_attention_heads: 8`
- `ff_multiplier: 4`
- `max_rollout_horizon: 128`
- `state_loss: "mse"`
- `reward_output_dim: 1`
- `state_loss_weight: 1.0`
- `reward_loss_weight: 1.0`
- `use_residual_transition: true`

#### Action decoder

- `latent_dim: 1536`
- `action_dim: 7`
- `hidden_dim: 512`
- `num_layers: 2`
- `dropout: 0.1`

#### Backbone config

- `model_path: null`
- `device: "cuda"`
- `dtype: "float16"`
- `freeze: false`

### C.3 Notes on an important internal inconsistency

The repository currently contains **two different latent spaces**:

- a **compact latent action space** of size `32` in the latent encoder, and
- a **feature-space world-model state** of size `1536` in the world model and decoder path.

This should be explicitly clarified in the final appendix because it is easy for reviewers to confuse “latent action” with the feature-space latent used by the world model.

**TODO for final paper**: add one paragraph and one diagram clarifying whether the 32-D latent is the stochastic action bottleneck and the 1536-D state is the backbone-conditioned transition space.

### C.4 Backbone details

From `configs/backbone.yaml` and `reports/backbone_compatibility.md`:

- Preferred backbone: `qwen3.5-9b`
- Runtime strategy: `cached-shared-clip`
- Shared encoder: `clip-vit-large-patch14`
- Cache directory: `processed/backbone_cache/clip-vit-large-patch14`
- Feature normalization: `true`

Backbone compatibility notes already documented in:

- `reports/backbone_compatibility.md`

That report identifies:

- **Qwen3.5-9B** as the recommended vision-capable backbone
- **BAGEL-7B-MoT** as a usable alternative
- **Qwen3-8B** as text-only and therefore not suitable as a vision backbone

### C.5 What is currently called the “world model”

The repository currently uses `src/models/world_model.py` to model latent transitions in feature space and optionally predict reward. However, the current draft text and rollout implementations still need stronger alignment around the exact meaning of “world model” in the paper.

**Appendix note to include explicitly:**

- Input to the world model: latent feature state plus action-conditioned representation
- Output: next latent state (feature-space) and reward prediction
- Test-time fair path in the current draft: does **not** use future observation
- The reward-aware latent dynamics are present in training and diagnostics, but the current repository still needs a clearer appendix explanation connecting these to final deployment-time behavior

---

## D. Training Objectives and Losses

### D.1 Draft-level paper objective (from `src/reports/neurips_draft_sections.tex`)

The current draft already states the following loss decomposition:

- Main action prediction loss:
  \( \mathcal{L}_{\mathrm{act}} = \lVert \hat a_t - a_t \rVert_2^2 \)
- World-model transition loss:
  \( \mathcal{L}_{\mathrm{wm}} = \lVert \hat z_{t+1} - z_{t+1}^{\star} \rVert_2^2 \)
- Refined action loss:
  \( \mathcal{L}_{\mathrm{ref\text{-}act}} = \lVert \tilde a_t - a_t \rVert_2^2 \)
- Latent refinement consistency loss:
  \( \mathcal{L}_{\mathrm{ref\text{-}lat}} = \lVert \tilde z_t - z_{t+1}^{\star} \rVert_2^2 \)

### D.2 Code-level losses already visible in the repo

#### Latent encoder / VAE-style component

From `src/models/latent_encoder.py`:

- reconstruction loss (MSE)
- KL divergence term
- `kl_weight = 1.0`

#### World model

From `src/models/world_model.py` and config files:

- state prediction loss: `mse` by default
- reward prediction loss
- state and reward weights both currently set to `1.0`

#### Action decoder

From `src/models/action_decoder.py` and training scripts:

- action reconstruction / regression loss is MSE-based in the current training/eval path

### D.3 Missing details still worth adding in final appendix

Not fully documented yet in the repo:

- whether KL annealing is used
- whether action normalization is standardized per task or globally
- exact reward label semantics for reward scorer training in the current paper narrative
- whether scorer and decoder are fully joint-trained or partially staged in the latest experimental path

These should be marked as `TODO` in the final paper appendix if not finalized.

---

## E. Test-Time Refinement / Adaptation Details

### E.1 Current repository reality

The repository contains a **latent refinement module**:

- `src/models/latent_refinement.py`

with parameters including:

- `steps = 5`
- `lr = 1e-2`
- `reward_weight = 1.0`
- `observation_weight = 1.0`
- `latent_anchor_weight = 1e-2`
- `max_grad_norm = 10.0`
- `max_latent_shift = 5.0`

### E.2 Important appendix clarification

The current draft and repo state do **not** support calling the released fair-path evaluation a CEM-based test-time planner. The repository contains refinement-style logic, but the draft figure notes already say the current implementation is better described as **training-time latent refinement / test-time direct prediction**, not a fully realized CEM search path.

Therefore the appendix should explicitly separate:

- **implemented latent refinement logic** (repository fact)
- **desired test-time adaptation metrics** such as `RG`, `rho_rank`, and `LUM` (evaluation protocol target)

### E.3 Success-based evaluation metrics already formalized

The success-metric protocol is already written in:

- `src/reports/table5_success_metric_protocol.tex`

and includes:

- `SR`
- `SR_ID`
- `SR_OOD`
- `Delta_gap`
- `RG`
- `rho_rank`
- `LUM`

At the repository level, this is a protocol definition and partial implementation state, not yet a fully completed measured benchmark.

---

## F. Benchmark, Dataset, and Task Protocol

### F.1 Canonical project-internal task list

From `configs/data.yaml`, the configured core tasks are:

1. `grab_roller`
2. `place_a2b_left`
3. `stack_blocks_two`
4. `handover_block`
5. `open_laptop`
6. `adjust_bottle`
7. `beat_block_hammer`
8. `click_bell`
9. `dump_bin_bigbin`
10. `press_stapler`

### F.2 Split ambiguity that must be documented

The repository currently contains **two split conventions**:

- `configs/train.yaml` / `configs/data.yaml`: `train=0.9`, `val=0.1`
- `src/reports/neurips_draft_sections.tex`: task-stratified `70/15/15`

This is a critical appendix item to resolve before submission.

**TODO:** finalize and document one canonical split protocol, then update all relevant scripts and tables to match it.

### F.3 Dataset location and format

- canonical dataset root in project: `data/robotwin/dataset/`
- tasks stored in extracted demo directories such as:
  - `data/robotwin/dataset/click_bell/extracted/franka_clean_50/video/episode0.mp4`
- ACT conversion datasets created under:
  - `data/act_format/`
  - `data/act_format_click_bell/`
  - `data/act_format_press_stapler/`
  - `data/act_format_beat_block_hammer/`
  - `data/act_format_click_bell_multi/`
  - `data/act_format_click_bell_14d16a/`

### F.4 Action and state dimensions

Repository evidence currently contains multiple conventions depending on evaluation path:

- offline baseline path commonly uses `action_dim=7`
- data config declares `state_dim=14`
- RoboTwin rollout control path expects a full `16D` action vector for dual-arm + gripper style commands

This mismatch is one of the most important appendix items to document clearly.

### F.5 Success criteria

Success is not currently centralized in one lara-wm-only document, but the project-internal rollout pipeline uses native RoboTwin task success predicates through:

- `scripts/run_robotwin_success_eval.py`

and logs its outputs under the project’s own:

- `experiments/rollout_success*/`

**TODO:** add a markdown table in the appendix with one row per task and an English description of the task-success condition.

---

## G. Baseline Details

### G.1 Internal baselines

#### Direct Policy

Files:

- `src/baselines/direct_policy.py`
- `src/deploy/direct_policy_deploy.py`
- `configs/direct_policy.yaml`

Description:

- backbone features are mapped directly to action outputs
- no latent refinement branch
- no reward-aware latent dynamics

#### Latent No-Refine

Files:

- `src/baselines/latent_no_refine.py`
- `src/deploy/latent_no_refine_policy.py`
- `configs/latent_no_refine.yaml`

Description:

- latent representation is used
- no iterative refinement in deployment path

#### No-Reward WM

Files:

- `src/baselines/no_reward_wm.py`

Description:

- transition modeling without reward-aware head

### G.2 External baselines already integrated into project results

#### ACT

Measured offline result file:

- `experiments/results/act_original_results.json`

Training / reproduction scripts:

- `scripts/train_eval_act_original.py`
- `scripts/convert_robottwin_to_act.py`

#### Diffusion Policy

Measured offline result file:

- `experiments/results/diffusion_policy_results.json`

Training scripts:

- `scripts/train_eval_diffusion_policy.py`
- `scripts/train_robottwin_diffusion_policy.py`

#### OpenVLA

Measured probe result files:

- `experiments/results/openvla_eval_1ep.json`
- `experiments/results/openvla_eval_3ep.json`

#### UniVLA

Measured result files:

- `experiments/results/univla_direct_1ep.json`
- `experiments/results/univla_direct_3ep.json`
- `experiments/results/univla_latent_decoder_2tr1val_per_task_3ep_test.json`
- `experiments/results/univla_latent_decoder_2tr1val_per_task_3ep_train.json`

---

## H. Hyperparameters and Training Details

### H.1 Verified training defaults from `configs/train.yaml`

| Hyperparameter | Value |
|---|---:|
| Batch size | 8 |
| Num workers | 4 |
| Num epochs | 100 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Gradient clip norm | 1.0 |
| Warmup steps | 1000 |
| Scheduler | cosine |
| Min LR | 1e-6 |

### H.2 Backbone defaults

| Item | Value |
|---|---|
| Preferred backbone | `qwen3.5-9b` |
| Shared encoder | `clip-vit-large-patch14` |
| Runtime strategy | `cached-shared-clip` |
| Device | `cuda` |
| Dtype | `float16` |
| Normalize features | `true` |

### H.3 Checkpointing defaults

From `configs/train.yaml`:

- save every 10 epochs
- save best checkpoint
- metric for best: `total_loss`
- keep 3 checkpoints

---

## I. Additional Quantitative Results Already Present in the Project

### I.1 Main offline paper-safe results

From `experiments/results/real_training_results.md`:

| Model | action_mse | action_mae | action_r2 |
|---|---:|---:|---:|
| direct_policy | 0.1961 ± 0.0029 | 0.2520 ± 0.0064 | 0.2292 ± 0.0116 |
| lara-wm | 0.1785 ± 0.0053 | 0.2391 ± 0.0065 | 0.2983 ± 0.0207 |
| latent_no_refine | 0.1933 ± 0.0026 | 0.2518 ± 0.0015 | 0.2400 ± 0.0103 |
| no_reward_wm | 0.1937 ± 0.0044 | 0.2501 ± 0.0012 | 0.2385 ± 0.0173 |

### I.2 Current external offline baseline summary

From `src/reports/report.md`:

| Model | Action MSE | Action MAE | Action R2 | Notes |
|---|---:|---:|---:|---|
| ACT | 0.0083 | 0.0405 | 0.9533 | offline reproduction complete |
| Diffusion Policy | 0.0077 | 0.0541 | 0.9535 | offline reproduction complete |
| UniVLA + latent decoder | 0.0116 | 0.0803 | 0.9344 | public backbone + RoboTwin decoder |
| OpenVLA (3 ep probe) | 0.8041 | 0.7129 | 0.6464 | zero-shot probe |
| UniVLA direct-action (3 ep probe) | 2.4561 | 1.3092 | very negative | weak direct-action probe |

### I.3 Success-rate status currently in project

From `src/reports/report.md`:

- native success evaluation entrypoint implemented: `scripts/run_robotwin_success_eval.py`
- smoke SR already measured on `grab_roller` under `demo_clean`
- current smoke values for:
  - `lara_wm`
  - `direct_policy`
  - `latent_no_refine`
  are all `0.00`

### I.4 Existing success-rate table artifacts

- `src/reports/table6_success_smoke.tex`
- `src/reports/table7_success_partial_progress.tex`

These should be treated as implementation/debugging artifacts, not final paper tables, unless the native rollout path is stabilized.

---

## J. Failure Analysis and Qualitative Evidence

### J.1 Existing project-internal failure evidence

Useful files:

- `reports/robotwin_smoke.log`
- `reports/robotwin_adjust_retry.log`
- `reports/robotwin_serial_sweep.log`
- `reports/robotwin_lara_multitask.log`
- `reports/robotwin_direct_multitask.log`
- `reports/robotwin_latent_multitask.log`

### J.2 Qualitative figures already prepared

- `src/reports/figure_snippets.tex`
- `src/reports/figures/paper_cases_v2/`
- `src/reports/figures/paper_cases_v3/`

Representative images generated in-project include:

- `click_bell_triptych_clean.png`
- `press_stapler_triptych_clean.png`
- `grab_roller_triptych_clean.png`
- `click_bell_single_clean.png`
- `adjust_bottle_single_clean.png`

These can be used in Appendix qualitative sections.

### J.3 Important current interpretation

The current repository evidence supports the following narrow conclusion:

- offline action prediction results are measurable and comparatively stable,
- but native task-success remains the unresolved bottleneck,
- and current zero-SR behavior is increasingly consistent with policy/control weakness rather than a pure evaluation wiring bug.

---

## K. Compute, Resources, and Reproducibility

### K.1 What is explicitly present in the repo

- CUDA-based execution is assumed in configs
- multiple scripts define exact commands for training/evaluation
- pretrained asset paths are recorded in `configs/asset_manifest.yaml`
- backbone compatibility is documented in `reports/backbone_compatibility.md`

### K.2 What is still missing and should be added before submission

The repository does **not** yet provide a clean, final appendix-ready record of:

- GPU model(s)
- number of GPUs used per experiment
- CPU / RAM
- per-method training wall-clock time
- total GPU-hours
- token/API cost (if any applicable final pipeline still uses external APIs)
- full seed table for every reported experiment
- confidence intervals / significance procedure for every main table

These should remain explicit `TODO` items unless they are added to the repository.

### K.3 Reproducibility command skeletons

The repository already implies the following command families:

```bash
python scripts/train_lara_wm.py
python scripts/train_and_eval.py
python scripts/train_eval_diffusion_policy.py
python scripts/train_eval_act_original.py
python scripts/run_robotwin_success_eval.py --baseline <name> --task-name <task>
```

**TODO:** add one exact, final command block per paper table into the appendix after the final experiment matrix is frozen.

---

## L. Limitations (Current Repository-Backed Version)

The repository currently supports the following honest limitations section:

1. The offline benchmark is substantially more mature than the native success-rate evaluation.
2. Current success-based evaluation infrastructure exists, but the strong non-zero policy result is still incomplete.
3. Multiple paths currently coexist for action/state dimensionality (7D, 14D, 16D), which should be rationalized in the final paper.
4. The project contains multiple backbone/runtime paths, and the final submission should fix one canonical configuration.
5. Native success may depend strongly on policy embodiment and execution details, so offline action quality alone is insufficient evidence of control robustness.

---

## M. Safety, Broader Impacts, and Asset Usage

### M.1 Broader impacts and safety

Repository evidence supports the following broad points:

- potential benefit: more robust robot action modeling under task variability and domain shift
- potential risk: reward/model mismatch may lead to unsafe or non-productive actions in execution
- deployment risk: test-time optimization or latent search can amplify model calibration errors if not constrained
- practical mitigation: bounded action outputs, rollout validation, simulator-first evaluation, and explicit native success testing

### M.2 Asset and dependency usage

The project explicitly depends on:

- RoboTwin data and assets (stored locally under `data/robotwin/dataset/` and related asset manifests)
- external baseline integrations under `third_party/`:
  - `act`
  - `diffusion_policy`
  - `openvla`
  - `univla`

**TODO:** add explicit license names and usage terms for each of these in the final paper appendix.

---

## N. LLM / Foundation-Model Usage Clarification

The repository currently contains multiple large vision-language / foundation-model backbones and third-party integrations, but the final appendix should clearly separate:

- what is part of the **core proposed method**,
- what is part of **baseline reproduction**, and
- what is part of **backbone infrastructure**.

At minimum, the final appendix should explicitly clarify:

- whether the core LaRA-WM contribution depends on an LLM-like planner (current evidence suggests no)
- which language/vision encoders are frozen versus trainable
- whether any external API or closed model is used in final reported experiments

---

## O. Minimum Must-Fill Items Before Submission

If time is limited, the highest-priority appendix items to complete are:

1. One canonical task list with exact train/val/test counts.
2. One canonical split definition (resolve 90/10 vs 70/15/15).
3. One canonical description of the world model and latent spaces.
4. A complete baseline table with fair training-budget notes.
5. Mean ± std / CI for all main reported results.
6. Native success-rate protocol details with explicit caveats.
7. GPU hardware, training times, and total compute.
8. Exact success criteria per RoboTwin task.
9. Explicit statement of what is measured versus aspirational in the current project state.
10. One clean reproducibility block with commands and config paths.

---

## P. Recommended Appendix File Layout for the Paper

```text
Appendix A. Full Method Details
Appendix B. Test-Time Refinement and Deployment Protocol
Appendix C. RoboTwin Benchmark, Tasks, and Success Criteria
Appendix D. Baseline Implementations
Appendix E. Hyperparameters and Training Details
Appendix F. Additional Quantitative Results
Appendix G. Failure Analysis and Qualitative Cases
Appendix H. Compute, Reproducibility, and Release Notes
Appendix I. Limitations, Safety, and Asset Usage
```

---

## Q. Repository-Internal TODO Summary

The following appendix-critical information is **not yet fully available** inside `/data/alice/cjtest/lara-wm` and should be explicitly completed before submission:

- exact final task split counts per task
- exact success criteria table per task
- final compute budget and GPU-hours
- final statistical significance and confidence intervals for all main tables
- one canonical success-rate result table with a fully stabilized native policy path
- final clarification of whether deployment uses direct prediction, refinement, or search in the claimed method

---

## R. Referenced External Projects (for Appendix Context)

The user explicitly allowed using information from referenced projects that LaRA-WM builds on or compares against. The following external/local reference projects are already mirrored or vendored into the broader workspace and can be cited in the appendix for reproducibility context.

### R.1 RoboTwin benchmark reference

Reference location outside the main project:

- `/data/alice/cjtest/AgentCode_Baseline/RoboTwin/`

Useful benchmark-facing files:

- `README.md`
- `script/eval_policy.py`
- `task_config/`
- `policy/ACT/`
- `policy/DP/`
- `policy/openvla-oft/`
- `policy/RDT/`

Why it matters:

- defines native task-success rollout semantics
- exposes official policy wrappers and embodiment assumptions
- provides benchmark-side context for action dimensionality, planner usage, and success predicates

### R.2 ACT reference

Project-local integration:

- `third_party/act/`

Relevant files:

- `third_party/act/README.md`
- `third_party/act/config/config.py`
- `third_party/act/config/config_robottwin.py`
- `third_party/act/train.py`
- `third_party/act/evaluate.py`

Appendix usage:

- baseline description
- chunked action prediction details
- reference for task-conditioned imitation-style policy execution

### R.3 Diffusion Policy reference

Project-local integration:

- `third_party/diffusion_policy/`

Relevant files:

- `third_party/diffusion_policy/README.md`
- `third_party/diffusion_policy/media/teaser.png`
- `third_party/diffusion_policy/media/multimodal_sim.png`
- `third_party/diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py`

Appendix usage:

- baseline implementation context
- policy-family design reference
- useful visual assets for baseline background (if properly attributed)

### R.4 OpenVLA reference

Project-local integration:

- `third_party/openvla/`

Relevant files:

- `third_party/openvla/README.md`
- `third_party/openvla/scaffolding/README.md`
- `third_party/openvla/scaffolding/run_eval.py`
- `third_party/openvla/scaffolding/run_inference.py`
- `third_party/openvla/scaffolding/compute_stats.py`

Appendix usage:

- zero-shot VLA probe details
- local statistics fallback notes
- deployment caveats for foundation-model baselines

### R.5 UniVLA reference

Project-local integration:

- `third_party/univla/`

Relevant files:

- `third_party/univla/README.md`
- `third_party/univla/docs/real-world-deployment.md`
- `third_party/univla/scaffolding/README.md`
- `third_party/univla/scaffolding/train_latent_decoder.py`
- `third_party/univla/scaffolding/run_eval.py`

Appendix usage:

- latent-decoder baseline details
- deployment assumptions for foundation-model policies
- latent action pretraining context

---

## S. Reusable Visual Assets and Candidate Figure Sources

### S.1 Project-internal paper figures (preferred)

Already saved under:

- `src/reports/figures/paper_cases_v3/`

Current files:

- `click_bell_triptych_clean.png`
- `press_stapler_triptych_clean.png`
- `grab_roller_triptych_clean.png`
- `click_bell_single_clean.png`
- `adjust_bottle_single_clean.png`

These are the current preferred paper figures because they are already curated inside the project and do not depend on external directories at paper-writing time.

### S.2 Earlier qualitative figures (secondary)

Located under:

- `src/reports/figures/paper_cases_v2/`

Current files:

- `bad_adjust_bottle_clean.png`
- `bad_click_bell_clean.png`
- `good_click_bell_clean.png`
- `good_grab_roller_clean.png`
- `good_press_stapler_clean.png`

### S.3 Raw demonstration videos for additional qualitative panels

Canonical internal source directory:

- `data/robotwin/dataset/`

Recommended tasks for qualitative visuals:

- `grab_roller`
- `click_bell`
- `press_stapler`
- `adjust_bottle`
- `beat_block_hammer`

Representative example files:

- `data/robotwin/dataset/grab_roller/extracted/franka_clean_50/video/episode0.mp4`
- `data/robotwin/dataset/click_bell/extracted/franka_clean_50/video/episode0.mp4`
- `data/robotwin/dataset/press_stapler/extracted/franka_clean_50/video/episode0.mp4`
- `data/robotwin/dataset/adjust_bottle/extracted/franka_clean_50/video/episode0.mp4`
- `data/robotwin/dataset/beat_block_hammer/extracted/franka_clean_50/video/episode0.mp4`

### S.4 External visual assets worth referencing (with attribution)

#### Diffusion Policy assets

- `third_party/diffusion_policy/media/teaser.png`
- `third_party/diffusion_policy/media/multimodal_sim.png`

These are useful for baseline-family background or appendix-side baseline illustrations, but should be clearly attributed and are less preferable than project-generated figures for the main paper.

#### UniVLA assets

Located under:

- `third_party/univla/assets/`

Notable files:

- `univla-teaser_new.png`
- `teaser_univla.png`
- `real-world-exp_1.png`
- `libero_latent_action_acc.png`
- `libero_action_loss.png`
- `latent-action-pretraining.png`

These are useful as reference/context for appendix-side related work or baseline background, not as primary evidence for LaRA-WM's own claims.

### S.5 Quantitative results that should be visualized as figures

The repository already contains enough structured data to generate the following figure types from project-local results:

1. **Main offline comparison bar chart**
   - source: `experiments/results/real_training_results.json`
   - recommended metrics: action MSE, action MAE, action R2

2. **Per-task performance heatmap**
   - source: `experiments/results/real_training_results_per_task.md`
   - recommended metric: per-task action R2

3. **Expanded external-baseline comparison figure**
   - source: `src/reports/table3_expanded_offline_comparison.tex`
   - highlight: ACT / Diffusion Policy / UniVLA vs LaRA-WM family

4. **Zero-shot VLA probe figure**
   - source: `src/reports/table4_zero_shot_vla_probe.tex`
   - highlight: OpenVLA and UniVLA direct-action behavior

5. **Success protocol / rollout progress figure**
   - source: `src/reports/table5_success_metric_protocol.tex`
   - optional appendix visualization only, until native success results are stabilized

---

## T. Practical Recommendation for Final Paper Assembly

If the appendix and figures are assembled today, the recommended priority order is:

### Main paper figures

1. `src/reports/figures/paper_cases_v3/click_bell_triptych_clean.png`
2. `src/reports/figures/paper_cases_v3/press_stapler_triptych_clean.png`
3. `src/reports/figures/paper_cases_v3/grab_roller_triptych_clean.png`

### Main paper tables

1. `src/reports/table3_expanded_offline_comparison.tex`
2. `src/reports/table4_zero_shot_vla_probe.tex`

### Appendix-only figures/tables

1. `src/reports/table5_success_metric_protocol.tex`
2. `src/reports/table6_success_smoke.tex`
3. `src/reports/table7_success_partial_progress.tex`
4. `src/reports/figures/paper_cases_v2/*`
5. externally attributed baseline assets from `third_party/` if needed
