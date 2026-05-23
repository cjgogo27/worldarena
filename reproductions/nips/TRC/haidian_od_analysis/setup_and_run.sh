#!/bin/bash
# 海淀区OD流量分析 - 环境安装和运行脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "海淀区OD流量分析系统 - 环境配置"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查Python版本
echo ""
echo "检查Python版本..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

# 创建虚拟环境
echo ""
echo "创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ 虚拟环境创建成功"
else
    echo "✓ 虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "安装依赖包..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ 环境配置完成！"
echo "=========================================="

# 询问是否立即运行
echo ""
read -p "是否立即运行分析程序? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=========================================="
    echo "开始运行分析..."
    echo "=========================================="
    python main.py
else
    echo ""
    echo "稍后可使用以下命令运行:"
    echo "  source venv/bin/activate"
    echo "  python main.py"
fi

echo ""
echo "完成！"
