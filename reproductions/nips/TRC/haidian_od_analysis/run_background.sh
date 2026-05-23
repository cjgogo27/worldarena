#!/bin/bash
# 海淀区OD分析后台运行脚本
# 功能: 使用screen在后台运行main.py，并将日志保存到文件

# 配置参数
PROJECT_DIR="/data/alice/cjtest/TRC/haidian_od_analysis"
LOG_DIR="${PROJECT_DIR}/logs"
SCREEN_NAME="haidian_od_analysis"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 检查是否已有同名screen会话
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "警告: Screen会话 '${SCREEN_NAME}' 已存在"
    echo "请使用以下命令查看状态:"
    echo "  screen -r ${SCREEN_NAME}  # 进入会话"
    echo "  screen -ls                # 查看所有会话"
    exit 1
fi

# 启动后台任务
echo "========================================"
echo "海淀区OD分析 - 后台运行"
echo "========================================"
echo "Screen会话名: ${SCREEN_NAME}"
echo "日志文件: ${LOG_FILE}"
echo "开始时间: $(date)"
echo "========================================"

# 使用screen在后台运行
screen -dmS "${SCREEN_NAME}" bash -c "
    cd ${PROJECT_DIR}
    echo '开始时间: \$(date)' > ${LOG_FILE}
    echo '========================================' >> ${LOG_FILE}
    python main.py >> ${LOG_FILE} 2>&1
    echo '========================================' >> ${LOG_FILE}
    echo '结束时间: \$(date)' >> ${LOG_FILE}
    echo '退出码: \$?' >> ${LOG_FILE}
"

# 等待1秒确保screen启动
sleep 1

# 检查是否成功启动
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "✓ 后台任务已启动成功"
    echo ""
    echo "常用命令:"
    echo "  查看实时日志:   tail -f ${LOG_FILE}"
    echo "  进入screen会话: screen -r ${SCREEN_NAME}"
    echo "  退出screen会话: Ctrl+A then D (不要Ctrl+C，会终止程序)"
    echo "  查看所有会话:   screen -ls"
    echo "  终止任务:       screen -X -S ${SCREEN_NAME} quit"
    echo ""
    echo "预计运行时间: 2-4小时 (取决于数据量)"
else
    echo "✗ 后台任务启动失败"
    exit 1
fi
