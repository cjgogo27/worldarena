#!/bin/bash
# 完整的数据预处理和模型训练流程

SCRIPT_DIR="/data/alice/cjtest/TRC/Travel_Time"
LOG_DIR="${SCRIPT_DIR}/logs"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "===================================================================="
echo "行程时间预测完整流程"
echo "===================================================================="
echo "开始时间: $(date)"
echo ""

# 步骤1：数据预处理和插值
echo "步骤1: 数据预处理和插值..."
echo "日志: ${LOG_DIR}/step1_preprocess_${TIMESTAMP}.log"
screen -dmS step1_preprocess bash -c "
    source ~/.bashrc
    conda activate trc
    cd ${SCRIPT_DIR}
    python -u step1_preprocess_and_interpolate.py > ${LOG_DIR}/step1_preprocess_${TIMESTAMP}.log 2>&1
    echo 'Step 1 finished at:' \$(date) >> ${LOG_DIR}/step1_preprocess_${TIMESTAMP}.log
"

# 等待步骤1完成
echo "等待步骤1完成..."
while screen -ls | grep -q "step1_preprocess"; do
    sleep 5
    tail -n 3 ${LOG_DIR}/step1_preprocess_${TIMESTAMP}.log | head -n 1
done

echo ""
echo "步骤1完成！"
echo ""
sleep 2

# 步骤2：模型训练
echo "步骤2: 模型训练（无数据泄露）..."
echo "日志: ${LOG_DIR}/step2_train_${TIMESTAMP}.log"
screen -dmS step2_train bash -c "
    source ~/.bashrc
    conda activate trc
    cd ${SCRIPT_DIR}
    python -u step2_train_no_leak.py > ${LOG_DIR}/step2_train_${TIMESTAMP}.log 2>&1
    echo 'Step 2 finished at:' \$(date) >> ${LOG_DIR}/step2_train_${TIMESTAMP}.log
"

echo ""
echo "===================================================================="
echo "后台运行已启动"
echo "===================================================================="
echo ""
echo "查看进度:"
echo "  步骤2日志: tail -f ${LOG_DIR}/step2_train_${TIMESTAMP}.log"
echo "  Screen会话: screen -ls"
echo "  进入会话: screen -r step2_train"
echo ""
echo "查看结果:"
echo "  插值数据: ${SCRIPT_DIR}/od_flow_interpolated.csv"
echo "  评估指标: ${SCRIPT_DIR}/no_leak_evaluation.csv"
echo "  预测结果: ${SCRIPT_DIR}/no_leak_predictions.csv"
echo "  可视化: ${SCRIPT_DIR}/no_leak_results.png"
echo "===================================================================="
