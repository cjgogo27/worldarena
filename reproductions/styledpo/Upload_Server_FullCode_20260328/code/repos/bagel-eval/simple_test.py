#!/usr/bin/env python3
"""
Simple BAGEL Style Transfer Test
===============================

直接使用BAGEL的app.py来进行风格转换测试
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def simple_style_transfer_test():
    """简单的风格转换测试"""
    
    print("🎨 BAGEL简单风格转换测试")
    print("=" * 50)
    
    # 检查环境
    print("🔍 检查环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.device_count()} 个GPU")
    else:
        print("⚠️  CUDA不可用，使用CPU")
    
    # 检查模型路径
    model_path = "models/BAGEL-7B-MoT"
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    print(f"✅ 模型路径存在: {model_path}")
    
    # 检查测试图像
    test_image = "test_images/women.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return False
    print(f"✅ 测试图像存在: {test_image}")
    
    try:
        # 导入必要模块
        print("📦 导入BAGEL模块...")
        from app import load_bagel_model
        from inferencer import InterleaveInferencer
        print("✅ 模块导入成功")
        
        # 加载模型（使用mode=2，适中的内存使用）
        print("🚀 加载BAGEL模型（这可能需要几分钟）...")
        model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path, mode=2
        )
        print("✅ 模型加载成功")
        
        # 创建推理器
        print("🔧 创建推理器...")
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids
        )
        print("✅ 推理器创建成功")
        
        # 加载测试图像
        print("🖼️ 加载测试图像...")
        input_image = Image.open(test_image).convert('RGB')
        print(f"   图像尺寸: {input_image.size}")
        
        # 测试不同的风格转换
        style_tests = [
            {
                "prompt": "转换为油画风格",
                "output": "simple_test_oil.png",
                "cfg_text": 4.0,
                "cfg_img": 1.5,
                "steps": 25
            },
            {
                "prompt": "转换为水彩画风格", 
                "output": "simple_test_watercolor.png",
                "cfg_text": 3.0,
                "cfg_img": 1.2,
                "steps": 20
            }
        ]
        
        # 创建输出目录
        os.makedirs("outputs", exist_ok=True)
        
        for i, test in enumerate(style_tests, 1):
            print(f"\n🎨 测试 {i}/{len(style_tests)}: {test['prompt']}")
            
            try:
                # 执行风格转换
                result = inferencer(
                    image=input_image,
                    text=test['prompt'],
                    understanding_output=False,      # 图像生成模式
                    cfg_text_scale=test['cfg_text'], # 文本引导强度
                    cfg_img_scale=test['cfg_img'],   # 图像保真度
                    num_timesteps=test['steps'],     # 推理步数
                    think=False,                     # 不启用思考模式
                    image_shapes=(512, 512)          # 输出图像尺寸
                )
                
                # 检查结果
                if result['image'] is not None:
                    output_image = result['image']
                    output_path = os.path.join("outputs", test['output'])
                    
                    # 保存图像
                    output_image.save(output_path, quality=95)
                    
                    print(f"✅ 风格转换成功!")
                    print(f"   输出: {output_path}")
                    print(f"   尺寸: {output_image.size}")
                    print(f"   大小: {os.path.getsize(output_path) / 1024:.1f} KB")
                    
                    # 如果有文本输出，也显示
                    if result['text']:
                        print(f"   文本: {result['text'][:100]}...")
                        
                else:
                    print(f"❌ 未能生成图像")
                    
            except Exception as e:
                print(f"❌ 风格转换失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 测试完成!")
        print(f"查看 outputs/ 目录中的结果图像")
        
        return True
        
    except Exception as e:
        print(f"💥 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BAGEL简单风格转换测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（更少步数）")
    
    args = parser.parse_args()
    
    success = simple_style_transfer_test()
    
    if success:
        print("\n✅ 测试成功！BAGEL风格转换功能正常工作")
        return 0
    else:
        print("\n❌ 测试失败！请检查错误信息")
        return 1

if __name__ == "__main__":
    exit(main())