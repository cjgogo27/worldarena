#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ABLATION_DIR="${PROJECT_ROOT}/experiments/ablation"
VARIANTS="no_skill random_skill shuffled_order atomic_only composed_only"
SCRIPT="${PROJECT_ROOT}/scripts/fill_ablation_table.py"

echo "Watching for ablation completion..."
while true; do
    all_done=true
    for v in $VARIANTS; do
        if [ ! -f "$ABLATION_DIR/$v/latest_metrics.json" ]; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        echo "All ablation variants complete! Running fill script..."
        python3 "$SCRIPT"
        break
    fi
    echo "$(date): Still waiting... $(ls $ABLATION_DIR/*/latest_metrics.json 2>/dev/null | wc -l)/5 done"
    sleep 120
done
