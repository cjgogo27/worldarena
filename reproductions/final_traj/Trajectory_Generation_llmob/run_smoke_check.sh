#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob"
cd "${BASE_DIR}"
python smoke_check.py --dataset "${1:-2019}"
