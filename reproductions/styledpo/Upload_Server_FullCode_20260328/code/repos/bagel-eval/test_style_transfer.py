#!/usr/bin/env python3
"""
测试BAGEL风格转换功能的示例脚本
"""

import os
import sys
from pathlib import Path

def test_style_transfer():
    """测试风格转换功能"""
    
    # 检查测试图像是否存在
    test_images = [
        "test_images/meme.jpg",
        "test_images/octupusy.jpg", 
        "test_images/women.jpg"
    ]
    
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ 未找到测试图像，请确保test_images目录中有图像文件")
        return False
    
    # 使用第一个可用的测试图像
    input_image = available_images[0]
    print(f"✅ 使用测试图像: {input_image}")
    
    # 风格转换示例
    style_examples = [
        {
            "prompt": "转换为油画风格，艺术感强烈，色彩浓郁",
            "output": "output_oil_painting.png"
        },
        {
            "prompt": "转换为水彩画风格，柔和淡雅，水墨效果",
            "output": "output_watercolor.png"
        },
        {
            "prompt": "转换为卡通风格，可爱有趣，色彩鲜艳",
            "output": "output_cartoon.png"
        }
    ]
    
    print(f"\n🎨 开始测试风格转换功能...")
    print(f"输入图像: {input_image}")
    
    for i, example in enumerate(style_examples, 1):
        print(f"\n📝 测试 {i}/{len(style_examples)}: {example['prompt']}")
        
        # 构建命令
        cmd = [
            "python", "style_transfer.py",
            "--input_image", input_image,
            "--style_prompt", example['prompt'],
            "--output", example['output'],
            "--num_timesteps", "25",  # 减少步数以加快测试
            "--cfg_text_scale", "4.0",
            "--cfg_img_scale", "2.0",
            "--seed", "42"  # 固定种子保证可重复性
        ]
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 执行命令
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ 风格转换成功: {example['output']}")
                
                # 检查输出文件是否存在
                if os.path.exists(example['output']):
                    file_size = os.path.getsize(example['output'])
                    print(f"   文件大小: {file_size / 1024:.1f} KB")
                else:
                    print(f"⚠️  输出文件不存在: {example['output']}")
                    
            else:
                print(f"❌ 风格转换失败")
                print(f"   错误输出: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 超时：风格转换耗时过长（>5分钟）")
        except Exception as e:
            print(f"❌ 执行错误: {e}")
    
    return True

def check_environment():
    """检查环境是否正确配置"""
    print("🔍 检查环境配置...")
    
    # 检查Python环境
    print(f"Python版本: {sys.version}")
    
    # 检查重要模块
    required_modules = [
        "torch", "PIL", "numpy", 
        "transformers", "accelerate"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 已安装")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}: 未安装")
    
    if missing_modules:
        print(f"\n⚠️  缺少必要模块: {', '.join(missing_modules)}")
        print("请在bagel2环境中运行: conda activate bagel2")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.device_count()} 个GPU")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU（速度较慢）")
    except:
        print("❌ 无法检查CUDA状态")
    
    # 检查模型文件
    model_path = "models/BAGEL-7B-MoT"
    required_files = [
        "llm_config.json",
        "vit_config.json", 
        "ae.safetensors",
        "ema.safetensors"
    ]
    
    print(f"\n📁 检查模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请下载BAGEL模型到models/BAGEL-7B-MoT目录")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file}: {file_size / (1024**3):.2f} GB")
        else:
            missing_files.append(file)
            print(f"❌ {file}: 不存在")
    
    if missing_files:
        print(f"\n⚠️  缺少模型文件: {', '.join(missing_files)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 BAGEL风格转换测试脚本")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先解决上述问题")
        return 1
    
    print("\n" + "=" * 50)
    
    # 测试风格转换
    if not test_style_transfer():
        print("\n❌ 风格转换测试失败")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("\n💡 使用提示:")
    print("1. 查看生成的图像文件")
    print("2. 调整参数以获得更好的效果")
    print("3. 尝试不同的风格提示词")
    
    return 0

if __name__ == "__main__":
    exit(main())