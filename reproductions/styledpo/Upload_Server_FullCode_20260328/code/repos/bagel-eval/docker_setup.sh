#!/bin/bash

# BAGEL Docker 构建和运行脚本
# 解决 flash-attn 安装问题的 Docker 方案

set -e

echo "BAGEL Docker 安装和运行脚本"
echo "============================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查NVIDIA Docker是否安装
if ! docker run --rm --gpus all nvidia/cuda:11.0-runtime-ubuntu20.04 nvidia-smi &>/dev/null; then
    echo "⚠️  警告：NVIDIA Docker 运行时未正确配置"
    echo "请确保已安装 nvidia-docker2 和 nvidia-container-runtime"
fi

# 询问用户选择Python版本
echo "请选择Python版本："
echo "1) Python 3.11 (推荐，兼容性更好)"
echo "2) Python 3.10 (原始要求)"
echo "3) 使用官方预构建镜像（推荐，最快）"
read -p "请输入选择 [1-3]: " version_choice

case $version_choice in
    1)
        DOCKERFILE="Dockerfile"
        IMAGE_NAME="bagel:py311"
        echo "✅ 选择了 Python 3.11 版本"
        ;;
    2)
        DOCKERFILE="Dockerfile.py310"
        IMAGE_NAME="bagel:py310"
        echo "✅ 选择了 Python 3.10 版本"
        ;;
    3)
        echo "✅ 使用官方预构建镜像"
        # 直接拉取预构建镜像
        echo "🔄 拉取预构建的BAGEL镜像..."
        docker pull davideuler/bagel:latest || {
            echo "❌ 预构建镜像拉取失败，请检查网络连接"
            echo "将使用本地构建方案..."
            DOCKERFILE="Dockerfile.py310"
            IMAGE_NAME="bagel:py310"
        }
        if docker images | grep -q "davideuler/bagel"; then
            IMAGE_NAME="davideuler/bagel:latest"
            echo "✅ 预构建镜像拉取成功！"
            run_container
            exit 0
        fi
        ;;
    *)
        echo "❌ 无效选择，默认使用 Python 3.10"
        DOCKERFILE="Dockerfile.py310"
        IMAGE_NAME="bagel:py310"
        ;;
esac

# 构建Docker镜像的函数
build_image() {
    echo ""
    echo "🔨 开始构建Docker镜像..."
    echo "这可能需要10-20分钟，请耐心等待..."
    echo "如果网络较慢，可以尝试使用代理或稍后重试"

    # 设置构建超时
    timeout 1800 docker build -f $DOCKERFILE -t $IMAGE_NAME . || {
        echo "❌ Docker镜像构建失败或超时"
        echo ""
        echo "🔧 可能的解决方案："
        echo "1. 检查网络连接是否稳定"
        echo "2. 设置Docker镜像代理"
        echo "3. 使用VPN或代理"
        echo "4. 稍后重试"
        exit 1
    }

    if [ $? -eq 0 ]; then
        echo "✅ Docker镜像构建成功！"
        return 0
    else
        echo "❌ Docker镜像构建失败"
        return 1
    fi
}

# 运行容器的函数
run_container() {
    echo ""
    echo "🚀 启动BAGEL容器..."
    echo "注意：请确保models目录中已下载模型文件"
    
    # 创建models目录（如果不存在）
    mkdir -p ./models
    
    # 检查models目录是否为空
    if [ -z "$(ls -A ./models)" ]; then
        echo "⚠️  警告：models目录为空"
        echo "请先下载BAGEL模型文件到models目录"
        echo "可以运行: python checkpoint.py"
    fi
    
    # 运行容器
    echo "🌐 启动Gradio界面，访问地址: http://localhost:7860"
    docker run -it --rm \
        --gpus all \
        -p 7860:7860 \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/test_images:/app/test_images \
        $IMAGE_NAME
}

# 构建镜像
build_image

# 询问是否立即运行
read -p "是否立即运行容器？[y/N]: " run_now

if [[ $run_now =~ ^[Yy]$ ]]; then
    run_container
else
    echo ""
    echo "📝 手动运行命令："
    echo "mkdir -p ./models"
    echo "docker run -it --rm --gpus all -p 7860:7860 -v \$(pwd)/models:/app/models -v \$(pwd)/test_images:/app/test_images $IMAGE_NAME"
fi

echo ""
echo "✅ 脚本执行完成！"