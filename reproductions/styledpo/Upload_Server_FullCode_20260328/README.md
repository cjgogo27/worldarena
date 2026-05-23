# Style-DPO 项目 - 上传服务器版本

**项目状态**：DPO 训练可正常工作，已解决显存问题  
**最后更新**：2025年12月11日  
**目标**：使用 DPO 优化 BAGEL 风格迁移模型

---

## 📋 项目进度总结

### ✅ 已完成（可直接接续）

1. **DPO 算法实现** ✓
   - 成功集成到 BAGEL 模型
   - 显存优化策略已实施（冻结部分层）
   - 可用 2 张 GPU 卡训练

2. **LoRA 微调** ✓
   - 已集成到代码
   - 参数效率提升 20 倍以上

3. **模型评估框架** ✓
   - CLIP 评分方法已测试（3组）
   - VLM 评估已完成（95.56% 准确率，14907/15600场景获胜）
   - 评估代码位于 `code/evaluate/`

4. **数据准备** ✓
   - 风格分类体系完善（40个二级类别）
   - BAGEL 生成的 560 张样式化图像
   - 风格提示词库已构建

5. **配置和解决方案** ✓
   - 显存分配策略文档
   - Qwen2 DPO 微调参考
   - 混合精度训练配置

### ⚠️ 进行中（需要继续完善）

1. **大规模正负样本对构造** 🔄
   - ❌ 需要编写自动匹配脚本
   - ❌ 需要使用 VLM 自动评分选择正负样本
   - 目前仅用 1 个样本对进行了测试（过拟合）

2. **完整训练循环** 🔄
   - ❌ 批量数据处理
   - ❌ 长时间训练稳定性验证

### 📊 关键指标与配置

**Qwen2 模型训练配置**（已验证可行）：
```
显存需求：46-52GB（双卡）
训练模式：DPO + LoRA
冻结模块：VAE、Embedding、所有LayerNorm、LM Head、MoE
训练投影层（7个）：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

DPO 超参：
  β = 0.1
  Learning Rate = 1e-5
  Warmup Steps = 10
```

**评估结果**：
```
VLM 总准确率：95.56%
总获胜次数：14907/15600
平均每风格获胜：372.68次
```

---

## 🗂️ 目录结构说明

```
Upload_Server/
├── code/                          # 核心代码
│   ├── train/                     # 训练相关脚本
│   │   ├── dpo_training.py        # DPO 训练主脚本
│   │   ├── lora_config.py         # LoRA 配置
│   │   └── qwen2_dpo.py           # Qwen2 DPO 参考
│   ├── data/                      # 数据处理脚本
│   │   ├── data_pairing.py        # ⭐ 待完成：样本对自动匹配
│   │   ├── build_preference_pairs.py  # 正负样本对构造
│   │   └── style_prompt_lib.py    # 风格提示词库
│   └── evaluate/                  # 评估脚本
│       ├── clip_eval.py           # CLIP 评分
│       ├── vlm_eval.py            # VLM 评分（Qwen3VL）
│       └── compute_metrics.py     # 指标计算
│
├── results/                       # 结果和日志
│   ├── checkpoints/               # 模型权重（需上传后更新）
│   ├── logs/                      # 训练日志
│   └── evaluations/               # 评估结果
│       ├── benchmark.json         # VLM 评估结果
│       └── clip_scores.csv        # CLIP 评分结果
│
├── resources/                     # 资源文件
│   ├── styles/                    # 风格图像和提示词
│   │   ├── style_categories.json  # 40 个风格分类
│   │   └── prompts/
│   └── datasets/                  # 公开数据集
│       ├── omnistyle/            # Omni-150K（1812 content 图）
│       ├── wikiart/              # WikiArt（20000+ 风格图）
│       └── artemis/              # Artemis 数据集
│
└── README.md                      # 本文件
```

---

## 🚀 接续步骤（优先级排序）

### **第 1 步**：修复启动脚本 ⭐ 立即需要
**位置**：`code/data/data_pairing.py`  
**任务**：编写自动匹配脚本生成正负样本对
```python
# 伪代码示意
def auto_pair_samples(generated_images, style_descriptions, vlm_model):
    """
    自动为生成的图像选择正负样本
    1. 使用 VLM 评分每张图像与风格描述的匹配度
    2. 选择匹配度最高的为 chosen
    3. 选择匹配度最低的为 rejected
    4. 生成 (chosen, rejected) 对，保存为 DPO 训练数据
    """
    pass
```

**预期输出格式**：
```json
[
  {
    "chosen": {"image_path": "...", "prompt": "..."},
    "rejected": {"image_path": "...", "prompt": "..."}
  }
]
```

---

### **第 2 步**：启动 DPO 训练 🔜 数据准备好后
**位置**：`code/train/dpo_training.py`  
**命令示例**：
```bash
# 双卡训练
torchrun --nproc_per_node=2 code/train/dpo_training.py \
    --model_id "bagel-model-id" \
    --train_data "path/to/preference_pairs.json" \
    --output_dir "./results/checkpoints" \
    --num_epochs 3 \
    --batch_size 4 \
    --beta 0.1
```

**关键注意**：
- ⚠️ 显存需求：52GB（双卡，建议使用 L40 或 A100）
- ⚠️ 冻结的层（VAE、Embedding 等）必须设定 `requires_grad=False`
- ⚠️ 使用 bfloat16 混合精度训练

---

### **第 3 步**：验证和评估 🔜 训练完成后
**位置**：`code/evaluate/`
```bash
# 1. CLIP 评估（快速）
python code/evaluate/clip_eval.py \
    --generated_images "results/checkpoint-xxx/images" \
    --output_csv "results/evaluations/clip_scores.csv"

# 2. VLM 评估（准确性高，但较慢）
python code/evaluate/vlm_eval.py \
    --generated_images "results/checkpoint-xxx/images" \
    --style_categories "resources/styles/style_categories.json" \
    --output_json "results/evaluations/vlm_results.json"

# 3. 计算总体指标
python code/evaluate/compute_metrics.py \
    --clip_results "results/evaluations/clip_scores.csv" \
    --vlm_results "results/evaluations/vlm_results.json"
```

---

## 💾 数据要求

### **训练数据** (需要준비)
- **样本对格式**：JSON 文件，包含选择的/被拒绝的图像对
- **样本数量**：建议 ≥ 1000 对（当前 1 对，只用于测速）
- **数据来源**：
  - BAGEL 生成的 560 张图像（已有）
  - Omni-150K content 图像（1812 张）

### **评估数据** (已准备)
- ✓ 40 种风格分类
- ✓ 每种风格的提示词库
- ✓ WikiArt 风格图像参考
- ✓ 预定义的评估提示词

---

## 🔧 依赖和环境

**Python 版本**：3.9+  
**GPU**：NVIDIA A100 或 L40（建议 2 张）  
**CUDA**：12.0+

**核心依赖**：
```
torch >= 2.0
transformers >= 4.36
diffusers >= 0.24
peft  # LoRA 支持
bitsandbytes  # 量化支持
pydantic
tqdm
```

**安装命令**：
```bash
pip install -r requirements.txt  # 待生成
```

---

## 🐛 已知问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| CUDA OOM | 显存不足 | ✓ 已解决：冻结 VAE/Embedding/LayerNorm，仅训练 7 个投影层 |
| 数据类型不匹配 | BFloat16 vs Float | ✓ 已解决：强制转换数据类型一致性 |
| 多分辨率图像处理 | 不能直接 resize | ✓ 已解决：token 级别独立平均，保留多分辨率优势 |
| LoRA 加载失败 | 维度不匹配 | ⚠️ 部分已修复，待完整测试 |
| 权重扁平化 | 模型分布加载问题 | ⚠️ 检查 `load_checkpoint_and_dispatch()` 使用方式 |

---

## 📞 重要联系人和参考

- **DPO 论文参考**：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **Diffusion-DPO 代码**：[SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO)
- **LoRA 微调参考**：通义千问 LoRA 实现
- **BAGEL 风格分类**：400 种二级风格，来自 Wikipedia 等公开资源

---

## 📅 下一步计划

**周期**：持续 2-4 周
1. **Week 1**：完成样本对自动匹配脚本，构造 1000+ 正负样本对
2. **Week 2**：运行完整 DPO 训练周期（3 个 epoch）
3. **Week 3**：完整评估和结果分析
4. **Week 4**：论文/报告撰写和模型发布

---

## ✨ 最后提醒

1. **显存管理**：在服务器运行前，务必确认有 2 张 ≥ 48GB 的 GPU
2. **数据备份**：checkpoints 大小约 4-6GB，务必定期备份
3. **日志监控**：训练日志每 100 steps 记录一次，持续监控
4. **中断恢复**：已实现 checkpoint save/load，可从中断点继续

---

**项目负责人**：[你的名字]  
**项目版本**：v2.0 (Ready for Server)  
**最后同步**：2025-12-11
