#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24"
cd "${BASE_DIR}"
python smoke_check.py --location "${1:-sf}"
