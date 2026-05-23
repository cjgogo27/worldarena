#!/bin/bash
# 无数据泄露版本行程时间预测后台运行脚本

SCRIPT_DIR="/data/alice/cjtest/TRC/Travel_Time"
LOG_DIR="${SCRIPT_DIR}/logs"
SCRIPT_NAME="predict_travel_time_no_leak.py"
LOG_FILE="${LOG_DIR}/no_leak_$(date +%Y%m%d_%H%M%S).log"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 激活conda环境并后台运行
screen -dmS travel_time_no_leak bash -c "
    source ~/.bashrc
    conda activate trc
    cd ${SCRIPT_DIR}
    python -u ${SCRIPT_NAME} > ${LOG_FILE} 2>&1
    echo 'Script finished at:' \$(date) >> ${LOG_FILE}
"

echo "===================================================================="
echo "无数据泄露版本已启动（后台运行）"
echo "===================================================================="
echo "Screen会话名: travel_time_no_leak"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "查看运行状态:"
echo "  tail -f ${LOG_FILE}"
echo "  screen -r travel_time_no_leak"
echo ""
echo "终止运行:"
echo "  screen -S travel_time_no_leak -X quit"
echo "===================================================================="
