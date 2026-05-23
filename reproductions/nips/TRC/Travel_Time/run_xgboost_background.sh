#!/bin/bash
# 行程时间预测XGBoost模型后台运行脚本

SCRIPT_DIR="/data/alice/cjtest/TRC/Travel_Time"
LOG_DIR="${SCRIPT_DIR}/logs"
SCRIPT_NAME="predict_travel_time_xgboost_v2.py"
LOG_FILE="${LOG_DIR}/xgboost_v2_$(date +%Y%m%d_%H%M%S).log"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 激活conda环境并后台运行
screen -dmS travel_time_xgboost bash -c "
    source ~/.bashrc
    conda activate trc
    cd ${SCRIPT_DIR}
    python -u ${SCRIPT_NAME} > ${LOG_FILE} 2>&1
    echo 'Script finished at:' \$(date) >> ${LOG_FILE}
"

echo "===================================================================="
echo "XGBoost行程时间预测模型已启动（后台运行）"
echo "===================================================================="
echo "Screen会话名: travel_time_xgboost"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "查看运行状态:"
echo "  screen -ls                          # 查看所有screen会话"
echo "  screen -r travel_time_xgboost       # 进入会话查看实时输出"
echo "  tail -f ${LOG_FILE}                 # 实时查看日志"
echo ""
echo "终止运行:"
echo "  screen -S travel_time_xgboost -X quit"
echo "===================================================================="
