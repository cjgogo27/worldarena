# BAGEL Style Transfer 使用指南

## 概述

本项目基于BAGEL多模态基础模型实现了图像风格转换功能。你可以输入一张图像和风格描述，生成具有指定风格的新图像。

## 环境要求

- Python 3.10
- NVIDIA GPU (推荐)
- CUDA 12.x
- 已安装的依赖项（见requirements.txt）

## 快速开始

### 1. 环境激活

```bash
conda activate bagel
```

### 2. 基本使用

```bash
python style_transfer.py \
    --input_image test_images/women.jpg \
    --style_prompt "watercolor painting style" \
    --output_path outputs/women_watercolor.jpg
```

### 3. 运行示例

```bash
python example_style_transfer.py
```

这会对test_images目录中的图像应用多种风格转换。

## 命令行参数

### 必需参数

- `--input_image`: 输入图像路径
- `--style_prompt`: 风格描述提示词
- `--output_path`: 输出图像路径

### 可选参数

- `--model_path`: BAGEL模型路径 (默认: models/BAGEL-7B-MoT)
- `--device`: 设备类型 (auto/cuda/cpu, 默认: auto)
- `--cfg_text_scale`: 文本引导强度 (默认: 7.0)
- `--cfg_image_scale`: 图像保真度 (默认: 1.2)
- `--num_timesteps`: 去噪步数 (默认: 50)
- `--seed`: 随机种子 (可选)

## 风格提示词示例

### 艺术风格
- "watercolor painting style"
- "oil painting style"
- "impressionist painting style"
- "Vincent van Gogh's Starry Night style"
- "Picasso cubist style"

### 媒介风格
- "pencil sketch style"
- "charcoal drawing style"
- "digital art style"
- "pixel art style"
- "comic book illustration style"

### 氛围风格
- "dreamlike fantasy style"
- "cyberpunk neon style"
- "vintage retro style"
- "minimalist modern style"

## 参数调优建议

### 高质量输出
```bash
python style_transfer.py \
    --input_image input.jpg \
    --style_prompt "your style" \
    --output_path output.jpg \
    --cfg_text_scale 8.0 \
    --cfg_image_scale 1.0 \
    --num_timesteps 100
```

### 快速预览
```bash
python style_transfer.py \
    --input_image input.jpg \
    --style_prompt "your style" \
    --output_path output.jpg \
    --cfg_text_scale 6.0 \
    --cfg_image_scale 1.5 \
    --num_timesteps 25
```

### 强化风格效果
```bash
python style_transfer.py \
    --input_image input.jpg \
    --style_prompt "your style" \
    --output_path output.jpg \
    --cfg_text_scale 10.0 \
    --cfg_image_scale 0.8 \
    --num_timesteps 75
```

## 参数说明

### cfg_text_scale (文本引导强度)
- 范围: 1.0 - 15.0
- 较高值 (7.0+): 更强烈的风格转换
- 较低值 (3.0-6.0): 更保守的转换

### cfg_image_scale (图像保真度)
- 范围: 0.5 - 2.0
- 较高值 (1.2+): 保持更多原图细节
- 较低值 (0.8-1.0): 允许更大的结构变化

### num_timesteps (去噪步数)
- 范围: 20 - 200
- 更多步数: 更高质量但更慢
- 较少步数: 更快但质量可能较低

## 输入图像建议

- **格式**: JPG, PNG
- **尺寸**: 512x512 到 2048x2048
- **内容**: 清晰、主体明确的图像效果更好
- **光照**: 良好光照的图像转换效果更佳

## 常见问题

### Q: 内存不足怎么办？
A: 尝试使用较小的图像尺寸或减少num_timesteps

### Q: 生成速度很慢？
A: 确保使用GPU，或减少num_timesteps参数

### Q: 风格转换效果不明显？
A: 增加cfg_text_scale值，减少cfg_image_scale值

### Q: 生成的图像质量不好？
A: 增加num_timesteps，调整cfg参数

## 文件结构

```
Bagel/
├── style_transfer.py           # 主脚本
├── example_style_transfer.py   # 示例脚本
├── STYLE_TRANSFER_GUIDE.md    # 本文档
├── test_images/               # 测试图像
├── style_transfer_outputs/    # 输出目录（自动创建）
└── models/BAGEL-7B-MoT/      # 模型文件
```

## 性能参考

在NVIDIA H800 GPU上的大致性能：

- **快速模式** (25 steps): ~30秒/图像
- **标准模式** (50 steps): ~60秒/图像  
- **高质量模式** (100 steps): ~120秒/图像

## 技术细节

本实现基于：
- **BAGEL**: 7B参数多模态扩散模型
- **Flash Attention**: 高效注意力机制
- **图像编辑管道**: 使用VAE编码器-解码器
- **CFG**: 分类器自由引导优化

## 许可证

本项目遵循Apache 2.0许可证。