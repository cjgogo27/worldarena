#!/bin/bash

# 快速本地安装flash-attn的脚本
# 使用已有的wheel文件

set -e

echo "🚀 BAGEL 快速安装脚本"
echo "===================="

# 激活conda环境
echo "📦 激活bagel环境..."
source ~/.bashrc
conda activate bagel

# 检查当前环境
echo "🔍 检查当前环境..."
python --version
nvidia-smi | grep "CUDA Version" || echo "CUDA信息获取失败"

# 检查是否已有wheel文件
if [ -f "flash_attn-2.5.8-cu122-cp310-linux_x86_64.whl" ]; then
    echo "✅ 找到现有的flash-attn wheel文件"
    echo "📦 强制安装本地wheel文件..."
    pip install flash_attn-2.5.8-cu122-cp310-linux_x86_64.whl --force-reinstall --no-deps
    
    # 如果失败，尝试使用manylinux标签
    if [ $? -ne 0 ]; then
        echo "❌ 标准安装失败，尝试重命名wheel文件..."
        cp flash_attn-2.5.8-cu122-cp310-linux_x86_64.whl flash_attn-2.5.8-py3-none-linux_x86_64.whl
        pip install flash_attn-2.5.8-py3-none-linux_x86_64.whl --force-reinstall --no-deps
    fi
else
    echo "⏬ 下载适配的flash-attn预编译包..."
    # 尝试多个下载源
    for url in \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.5.8+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    do
        echo "🔄 尝试下载: $url"
        wget -O flash_attn_wheel.whl "$url" && break
        echo "❌ 下载失败，尝试下一个..."
    done
    
    if [ -f "flash_attn_wheel.whl" ]; then
        echo "📦 安装下载的wheel文件..."
        pip install flash_attn_wheel.whl --force-reinstall --no-deps
        
        # 如果失败，尝试重命名
        if [ $? -ne 0 ]; then
            echo "❌ 安装失败，尝试绕过平台检查..."
            pip install flash_attn_wheel.whl --force-reinstall --no-deps --force-reinstall --break-system-packages
        fi
    else
        echo "❌ 下载失败，尝试直接pip安装..."
        pip install flash-attn==2.5.8 --no-build-isolation
    fi
fi

# 验证安装
echo "🔍 验证flash-attn安装..."
python -c "import flash_attn; print('✅ flash-attn版本:', flash_attn.__version__)" || {
    echo "❌ flash-attn安装失败"
    exit 1
}

# 安装其他依赖
echo "📦 安装其他依赖包..."
pip install -r requirements.txt

echo ""
echo "✅ 安装完成！"
echo "🚀 现在可以运行BAGEL了："
echo "   python app.py"