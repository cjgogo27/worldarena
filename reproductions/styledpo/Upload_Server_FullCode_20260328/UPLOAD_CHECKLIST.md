#!/bin/bash
# 文件清单和上传指南
# 生成时间：2025-12-11

## ===== 1. 需要从 Infer 目录复制的核心代码 =====

# 从 Infer/qwen-dpo-main 复制 DPO 训练框架
# 文件：
#   - DPO.py (Qwen2 DPO 实现参考)
#   - DPO.ipynb (训练笔记本)
#   位置→ code/train/qwen2_dpo.py (参考)

# 从 Infer/K-LoRA-main 复制 LoRA 实现
# 文件：
#   - train_dreambooth_lora_sdxl.py (LoRA 训练脚本)
#   - utils.py (工具函数)
#   位置→ code/train/lora_config.py (提取关键部分)

# 从 Infer/RB-Modulation-main 复制评估模块
# 文件：
#   - utils.py (VLM 特征提取)
#   位置→ code/evaluate/vlm_eval.py (参考)

# 从 Infer/Bagel/batch_style_transfer.py 复制批处理脚本
# 位置→ code/train/dpo_training.py (修改为 DPO)

## ===== 2. 需要创建的新文件 =====

### code/train/
# - dpo_training.py (主训练脚本，DPO + LoRA)
# - lora_config.py (LoRA 配置参数)
# - config.yaml (训练超参配置)
# - qwen2_dpo.py (参考实现，复制自 Infer)

### code/data/
# ⭐ data_pairing.py (自动样本匹配，需要新写)
# - build_preference_pairs.py (构造训练数据)
# - style_prompt_lib.py (风格提示词库)
# - dataloader.py (数据加载器)

### code/evaluate/
# - clip_eval.py (CLIP 评分脚本)
# - vlm_eval.py (VLM 评分脚本，Qwen3VL)
# - compute_metrics.py (指标计算)

### resources/
# - style_categories.json (40 种风格分类)
# - style_prompts.txt (详细提示词)

## ===== 3. 需要在服务器准备的资源 =====

### results/
# ✓ benchmarks/ (已有评估结果，可参考)
#   - Benchmark - 飞书云文档 (下载 JSON 和 CSV)
# ✗ checkpoints/ (待生成，运行训练后生成)
# ✗ logs/ (待生成)

### resources/datasets/
# ✓ omnistyle/ (1812 content 图像)
#   位置：Dataset/OmniStyle-150Ka/OmniStyle-150K/
# ✓ wikiart/ (风格参考图像)
#   位置：Wikiart/
# ✓ artemis/ (文本-图像对数据)
#   位置：Dataset/artemis_official_data/

## ===== 4. 上传清单 =====

# 必须上传（核心代码）：
[ ] Upload_Server/code/train/**/*.py
[ ] Upload_Server/code/data/**/*.py
[ ] Upload_Server/code/evaluate/**/*.py
[ ] Upload_Server/README.md
[ ] Upload_Server/requirements.txt (待生成)

# 应该上传（结果和配置）：
[ ] Upload_Server/resources/styles/style_categories.json
[ ] Upload_Server/resources/styles/style_prompts.txt
[ ] 训练配置文件示例

# 可选上传（参考文件）：
[ ] Infer/qwen-dpo-main/DPO.py (参考)
[ ] Infer/K-LoRA-main/train_dreambooth_lora_sdxl.py (参考)
[ ] Training logs and benchmarks (用于对比)

## ===== 5. 服务器数据准备清单 =====

# 需要在服务器端准备的数据集：
[*] Omni-150K content images (1812 张)
    命令：tar -xvf OmniStyle-150K.tar.part_*
    位置：/data/datasets/omnistyle/

[*] WikiArt 风格参考 (需要 2GB 左右)
    位置：/data/datasets/wikiart/

[*] Artemis 数据集 (文本描述)
    位置：/data/datasets/artemis/

## ===== 6. 关键文件大小估算 =====

# 代码文件：
Upload_Server/code/            ~50 MB (Python 脚本)
Upload_Server/resources/       ~100 MB (提示词库、配置)

# 模型权重（待生成）：
BAGEL base model:              ~4 GB
Qwen2 base model:              ~8 GB  
LoRA weights (微调后):          ~100 MB

# 数据文件：
content images (1812):         ~2 GB
style images reference:        ~1 GB
preference_pairs.json:         ~500 MB (1000+ 样本对)

# 结果文件：
Checkpoints (per epoch):       ~2 GB
Logs and metrics:              ~100 MB

# 总计需要空间：
~20-25 GB (如果使用本地模型)
~3-5 GB (如果使用 Hugging Face 在线模型)

## ===== 7. 环境配置检查清单 =====

# 在服务器上执行：
[ ] python --version  # >= 3.9
[ ] nvidia-smi  # CUDA 12.0+，查看 GPU
[ ] pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
[ ] pip install -r Upload_Server/requirements.txt

# 验证依赖：
[ ] python -c \"import torch; print(torch.__version__)\"
[ ] python -c \"from transformers import AutoModel\"
[ ] python -c \"from diffusers import DiffusionPipeline\"

## ===== 8. 首次运行步骤 =====

# 1. 下载/准备数据
[ ] tar -xvf OmniStyle-150K.tar.part_* -C /data/datasets/
[ ] unzip 或复制 WikiArt 数据到 /data/datasets/wikiart/

# 2. 生成样本对 (重要！)
[ ] python code/data/data_pairing.py \\
      --content_dir /data/datasets/omnistyle \\
      --output_file ./results/preference_pairs.json

# 3. 启动训练
[ ] torchrun --nproc_per_node=2 code/train/dpo_training.py \\
      --config config.yaml \\
      --train_data ./results/preference_pairs.json

# 4. 监控训练
[ ] tail -f results/logs/training.log
[ ] tensorboard --logdir ./results/logs/

# 5. 完成训练后评估
[ ] python code/evaluate/clip_eval.py --checkpoint ./results/checkpoints/best
[ ] python code/evaluate/vlm_eval.py --checkpoint ./results/checkpoints/best

## ===== 9. 常见问题快速诊断 =====

# Q: CUDA OOM
# A: 检查冻结层是否设置（VAE、Embedding 等）
# 命令: grep \"requires_grad\" code/train/dpo_training.py

# Q: 数据类型错误
# A: 检查数据加载器是否强制转换为 bfloat16
# 命令: grep \"torch.bfloat16\" code/train/dpo_training.py

# Q: 模型加载失败
# A: 检查 device_map 和 quantization 配置
# 命令: python -c \"from transformers import AutoModel; AutoModel.from_pretrained(...)\"

# Q: 样本对生成失败
# A: 检查 VLM API/本地模型是否可用
# 命令: python code/data/data_pairing.py --dry_run

## ===== 10. 备份和恢复 =====

# 定期备份 (每 12 小时)：
[ ] tar -czf backups/checkpoint-$(date +%Y%m%d-%H%M).tar.gz results/checkpoints/
[ ] tar -czf backups/logs-$(date +%Y%m%d-%H%M).tar.gz results/logs/

# 恢复训练：
[ ] tar -xzf backups/checkpoint-latest.tar.gz -C results/
[ ] python code/train/dpo_training.py --resume_from_checkpoint ./results/checkpoints/last

---
生成时间：2025-12-11
更新者：[你的名字]
版本：v2.0 - Server Ready
