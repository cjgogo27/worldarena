# LaRA-WM: Latent Robot Action World Model

A task-centric world model combining latent action learning with reward-conditioned prediction and test-time refinement. Built on top of RoboTwin for real robot manipulation tasks.

## Project Structure

```
lara-wm/
├── configs/           # Training and model configs
│   ├── asset_manifest.yaml    # Asset paths and validation commands
│   ├── train.yaml            # Training hyperparameters
│   ├── model.yaml            # Model architecture
│   └── data.yaml             # Dataset configuration
├── data/
│   └── robotwin/
│       └── dataset/           # Task datasets (10 manipulation tasks)
├── scripts/
│   ├── train_lara_wm.py       # Main training entrypoint
│   └── run_experiments.py     # Baseline comparison runner
├── src/
│   ├── models/               # Core model implementations
│   │   ├── latent_encoder.py  # VAE-based latent action encoder
│   │   ├── world_model.py    # GRU-based world model with reward head
│   │   └── action_decoder.py  # Latent-to-action decoder
│   ├── baselines/            # Ablation baselines
│   ├── backbone/              # VLM backbone adapter
│   ├── data/                  # Dataset readers and standardization
│   └── eval/                  # Evaluation harness
├── experiments/              # Checkpoints and results
└── reports/                   # Ablation tables and feasibility gates
```

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n lara-wm python=3.11 -y
conda activate lara-wm

# Install core dependencies
pip install torch torchvision
pip install transformers h5py numpy scikit-learn pyyaml
pip install tensorboard  # For logging

# Verify environment
python scripts/train_lara_wm.py --help
```

### Verify Asset Access

```bash
# Verify RoboTwin repository
ls /data/alice/cjtest/AgentCode_Baseline/RoboTwin

# Verify backbone models
ls /data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/Qwen3.5-9B/config.json
ls /data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/BAGEL-7B-MoT/config.json

# Verify dataset
ls /data/alice/cjtest/lara-wm/data/robotwin/dataset/
```

## Asset Paths

### Backbone Models

| Model | Path | Role |
|-------|------|------|
| Qwen3.5-9B | `/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/Qwen3.5-9B` | Primary VLM backbone |
| BAGEL-7B-MoT | `/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/BAGEL-7B-MoT` | Alternate backbone |
| Qwen3-8B | `/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B` | Text-only fallback |

### RoboTwin Baselines

| Baseline | Path |
|----------|------|
| ACT | `/data/alice/cjtest/AgentCode_Baseline/RoboTwin/policy/ACT` |
| Diffusion Policy | `/data/alice/cjtest/AgentCode_Baseline/RoboTwin/policy/DP` |
| OpenVLA-oft | `/data/alice/cjtest/AgentCode_Baseline/RoboTwin/policy/openvla-oft` |

### Datasets

Dataset root: `/data/alice/cjtest/lara-wm/data/robotwin/dataset`

Available tasks:
- `grab_roller`
- `place_a2b_left`
- `stack_blocks_two`
- `handover_block`
- `open_laptop`
- `adjust_bottle`
- `beat_block_hammer`
- `click_bell`
- `dump_bin_bigbin`
- `press_stapler`

Official dataset (HuggingFace): `TianxingChen/RoboTwin2.0`

## Training

### Basic Training

```bash
# Train with default config
python scripts/train_lara_wm.py

# Train with custom config
python scripts/train_lara_wm.py --config configs/train.yaml

# Resume from checkpoint
python scripts/train_lara_wm.py --resume experiments/lara_wm_default/best.pt
```

### Training Configuration

Key parameters in `configs/train.yaml`:

```yaml
# Training
num_epochs: 100
learning_rate: 0.0001
gradient_clip_norm: 1.0
scheduler: "cosine"

# Data
batch_size: 8
num_workers: 4
train_split: 0.9

# Model dimensions
latent_encoder:
  latent_dim: 32        # Latent action dimension
  action_dim: 7         # Robot action space

world_model:
  latent_dim: 1536     # Feature dimension
  architecture: "gru"   # Transition model
```

### Checkpoints

Checkpoints saved to: `/data/alice/cjtest/lara-wm/experiments/{experiment_name}/`

- `best.pt` - Best model by validation loss
- `checkpoint_epoch_*.pt` - Periodic checkpoints (every 10 epochs by default)

## Experiments

### Run Baseline Comparisons

```bash
# Run all baselines
python scripts/run_experiments.py

# Run specific baseline
python scripts/run_experiments.py --baseline direct_policy
python scripts/run_experiments.py --baseline latent_no_refine
python scripts/run_experiments.py --baseline no_reward_wm

# Run with custom parameters
python scripts/run_experiments.py --num-episodes 100 --max-steps 300
```

### Ablation Experiments

The baseline implementations in `src/baselines/`:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| `direct_policy` | Forward backbone to actions | Vanilla baseline |
| `latent_no_refine` | Latent encoder + world model, no refinement | Ablate refinement |
| `no_reward_wm` | World model without reward head | Ablate reward conditioning |

## Model Architecture

LaRA-WM consists of three learned components:

1. **Latent Encoder** (VAE): Maps robot actions to compact latent representations
2. **World Model** (GRU): Predicts latent state transitions conditioned on rewards
3. **Action Decoder**: Maps predicted latents back to robot actions

The system supports test-time refinement through iterative latent refinement.

## Results

See `reports/` for:
- `feasibility_gates.md` - Development milestone verification
- `table1_main_comparison.md` - Main results table
- `table2_ablation.md` - Ablation study results

## Environment Setup for RoboTwin (Optional)

If using RoboTwin baselines directly:

```bash
# RoboTwin requires Python 3.10 with sapien
conda create -n robotwin python=3.10 -y
conda activate robotwin

cd /data/alice/cjtest/AgentCode_Baseline/RoboTwin
bash script/_install.sh
```