# WorldArena Reproduction

Training and evaluation code for WorldArena leaderboard submissions, including reproductions of **ABoT**, **VideoX-Fun (Wan2.1 LoRA)**, **TravelReasoner**, and other mobility/video generation baselines.

## Repository Structure

```
├── videox_fun/                    # VideoX-Fun framework (Wan2.1 training & inference)
├── scripts/
│   ├── wan2.1/                   # Wan2.1 LoRA training + inference scripts
│   │   ├── train_lora.py         # LoRA training for Wan2.1 I2V
│   │   ├── train_distill.py      # Distillation training
│   │   ├── batch_predict_i2v_worldarena.py  # WorldArena batch inference
│   │   ├── build_manifests_for_instructions.py
│   │   └── run_*infer.sh         # Inference launchers
│   ├── worldarena/               # SeedVR refinement & evaluation pipeline
│   │   ├── run_seedvr.sh         # SeedVR batch refiner (GPU-aware, auto-recovery)
│   │   ├── launch_instructions1_seedvr.sh  # Launch instructions_1 SeedVR
│   │   ├── launch_instructions2_seedvr.sh  # Launch instructions_2 SeedVR
│   │   ├── launch_instructions1_eval.sh    # Launch instructions_1 evaluation
│   │   ├── launch_instructions2_eval.sh    # Launch instructions_2 evaluation
│   │   └── run_eval_pipeline.sh  # Full eval pipeline (Wan2.1 → SeedVR → Metrics)
│   └── *.py                      # Utility scripts
├── config/                       # Configuration files
├── abot/
│   ├── worldarena_abot_public/   # ABoT WorldArena submission code
│   └── ABot-PhysWorld/           # ABot PhysWorld reproduction (code only)
├── worldarena_wan/               # WorldArena Wan2.1 public evaluation code
├── reproductions/
│   ├── travel_reasoner/          # TravelReasoner (Plan-R1) reproduction
│   ├── styledpo/                 # Style-DPO v2.8 reproduction
│   ├── nips/                     # NIPS paper reproductions
│   │   ├── geoskill/             # GeoSkill
│   │   ├── skillgeo/             # SkillGeo
│   │   ├── TRC/                  # TRC
│   │   ├── TRE/                  # TRE
│   │   ├── NIPS_Mainbody/        # NIPS Mainbody (AutoGeo)
│   │   └── Mobility_Foundation/  # Mobility Foundation
│   ├── final_traj/               # FinalTraj mobility generation
│   ├── lara-wm/                  # Lara-WM world model
│   ├── physics_state_gen/        # Physics state generation lab
│   ├── star_vla/                  # StarVLA
│   └── agentcode_baselines/      # Agent code baselines
├── *.py                          # Top-level data processing & training scripts
└── .gitignore
```

## Key Scripts

### Data Preparation
- `prepare_wan_lora_training.py` - Prepare RobotWin dataset for Wan2.1 LoRA training
- `convert_robotwin_to_videox_fun.py` - Convert RobotWin data to VideoX-Fun format
- `convert_robotwin_to_wan.py` - Convert RobotWin data to Wan2.1 format
- `reformat_data_for_training.py` - Reformat training data
- `download_robotwin_*.py` - Download RobotWin dataset from HuggingFace
- `download_extract_convert_pipeline.py` - End-to-end data pipeline

### Training
- `scripts/wan2.1/train_lora.py` - LoRA fine-tuning for Wan2.1 I2V-14B-480P
- `scripts/wan2.1/train_distill.py` - Distillation training
- `scripts/wan2.1/train_distill_lora.py` - Distillation + LoRA

### Inference / Evaluation (WorldArena)
- `scripts/wan2.1/batch_predict_i2v_worldarena.py` - Batch inference for WorldArena
- `scripts/wan2.1/run_instructions_infer.sh` - Launch inference on test instructions
- `batch_inference_test_dataset.py` - Batch inference on test dataset
- `scripts/wan2.1/build_manifests_for_instructions.py` - Build manifests for WorldArena instruction sets

### SeedVR Refinement Pipeline
- `scripts/worldarena/run_seedvr.sh` - SeedVR batch refiner: watches an input directory, runs SeedVR2-EMA-3B inference, and collects outputs. Uses `pick_gpu()` for automatic GPU selection or pinned GPU mode. Designed to survive shell timeout via `setsid -f`.
- `scripts/worldarena/launch_instructions1_seedvr.sh` - Launcher for SeedVR refinement on instructions_1 test set (1000 videos, GPU 2).
- `scripts/worldarena/launch_instructions2_seedvr.sh` - Launcher for SeedVR refinement on instructions_2 test set (1000 videos, GPU 3).
- `scripts/worldarena/launch_instructions1_eval.sh` - Launcher for evaluation metrics on instructions_1 outputs.
- `scripts/worldarena/launch_instructions2_eval.sh` - Launcher for evaluation metrics on instructions_2 outputs.
- `scripts/worldarena/run_eval_pipeline.sh` - Full end-to-end evaluation pipeline: generates videos with Wan2.1 inference, refines with SeedVR, computes quality metrics, and generates VLM summaries.

The SeedVR refiner uses a 3.4B-parameter DiT model (SeedVR2-EMA-3B) with Euler sampler (1 step), running at 832×480 resolution and 24 FPS. Each video is 121 frames.

## Checkpoints & Data

- **Checkpoints / model weights** are excluded from this repo (see `.gitignore`)
- Training data: RobotWin dataset from HuggingFace
- Test data: WorldArena test_dataset (2026-03-06 release)

## Submission Format (Track 1)

```
modelname_test/       # 1000 videos from main instructions
modelname_test_1/     # 1000 videos from instructions_1
modelname_test_2/     # 1000 videos from instructions_2
model_README.md       # Model metadata
```

Each video: 480×832, 24fps, 121 frames, MP4 format.
