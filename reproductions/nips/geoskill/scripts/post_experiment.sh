#!/usr/bin/env bash
# post_experiment.sh — Run after main experiment completes
# Usage: bash scripts/post_experiment.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "POST-EXPERIMENT PIPELINE"
echo "$(date)"
echo "=================================================="

# 1. Verify main experiment results exist
echo ""
echo "[1/5] Checking main experiment results..."
METHODS="direct_vlm cot_vlm geocot georeasoner gre_multistage skill_conditioned img2loc_rag"
MISSING=0
for m in $METHODS; do
    if [ ! -f "experiments/full_100/$m/latest_metrics.json" ]; then
        echo "  MISSING: experiments/full_100/$m/latest_metrics.json"
        MISSING=1
    else
        echo "  OK: $m"
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some main experiment results are missing. Run the main experiment first."
    exit 1
fi

# 2. Run analyses
echo ""
echo "[2/5] Running region analysis..."
python scripts/analyze_regions.py \
    --results-dir experiments/full_100 \
    --methods $METHODS \
    2>&1 | tee experiments/full_100/region_analysis_log.txt

echo ""
echo "[3/5] Running error analysis..."
python scripts/analyze_errors.py \
    --results-dir experiments/full_100 \
    --methods $METHODS \
    2>&1 | tee experiments/full_100/error_analysis_log.txt

# 3. Generate figures and tables
echo ""
echo "[4/5] Generating figures and tables..."
python scripts/generate_figures.py \
    --results-dir experiments/full_100 \
    --ablation-dir experiments/ablation \
    --output-dir figures \
    2>&1 | tee figures/generation_log.txt

# 4. Print summary
echo ""
echo "[5/5] Summary of generated outputs..."
echo ""
echo "--- Main Results ---"
cat experiments/full_100/summary_metrics.json 2>/dev/null || echo "(not found)"
echo ""
echo "--- Figures ---"
ls -la figures/*.pdf 2>/dev/null || echo "(no figures yet)"
echo ""
echo "--- Tables ---"
ls -la figures/*.tex 2>/dev/null || echo "(no tables yet)"

echo ""
echo "=================================================="
echo "POST-EXPERIMENT PIPELINE COMPLETE"
echo "$(date)"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run ablation:  python scripts/run_ablation.py --config configs/full.yaml"
echo "  2. Regenerate figures after ablation: python scripts/generate_figures.py --results-dir experiments/full_100 --ablation-dir experiments/ablation --output-dir figures"
echo "  3. Compile paper: cd paper && tectonic neurips_2026.tex"
