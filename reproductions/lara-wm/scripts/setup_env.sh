#!/bin/bash
# LaRA-WM Project Setup Script
# Run this to set up the project environment

set -e

echo "=== LaRA-WM Project Setup ==="

# 1. Create conda environment (recommended)
echo "[1/4] Creating conda environment..."
conda create -n lara-wm python=3.11 -y
conda activate lara-wm

# 2. Install core dependencies
echo "[2/4] Installing core dependencies..."
pip install torch torchvision numpy scipy scikit-learn
pip install h5py zarr pandas
pip install transformers accelerate einops
pip install pillow opencv-python imageio
pip install tensorboard wandb tqdm pyyaml omegaconf
pip install pytest pytest-cov

# 3. Download RoboTwin dataset (optional - large)
echo "[3/4] RoboTwin dataset download..."
echo "To download official dataset, run:"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('TianxingChen/RoboTwin2.0', repo_type='dataset')\""
echo "Or use: bash collect_data.sh <task> <config> <gpu>"

# 4. Verify installation
echo "[4/4] Verifying installation..."
python -c "import torch; import transformers; import h5py; print('All packages OK')"

echo "=== Setup Complete ==="
echo "To activate: conda activate lara-wm"