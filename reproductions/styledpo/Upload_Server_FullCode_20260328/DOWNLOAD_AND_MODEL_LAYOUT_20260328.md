# 下载与模型放置说明（2026-03-28）

本说明用于服务器首轮部署，目标是：先把必须资源下载到统一路径，再开始 benchmark/pair/dpo 任务。

## 1. 推荐目录布局

项目根目录：
- `Upload_Server_FullCode_20260328/`

模型目录（建议）：
- `Upload_Server_FullCode_20260328/models/BAGEL-7B-MoT/`
- `Upload_Server_FullCode_20260328/models/clip-vit-large-patch14/`
- `Upload_Server_FullCode_20260328/models/<your-vlm-model>/`

结果目录：
- `Upload_Server_FullCode_20260328/results/setup/assets_manifest_20260328.json`

## 2. 一键下载脚本位置

- `code/setup/download_required_assets.py`

## 3. 运行命令（推荐）

在项目根目录执行：

```bash
python code/setup/download_required_assets.py \
  --output_root models \
  --bagel_repo ByteDance-Seed/BAGEL-7B-MoT \
  --clip_repo openai/clip-vit-large-patch14 \
  --vlm_repo Qwen/Qwen3.5-9B
```

如果是私有/受限模型：

```bash
export HF_TOKEN=your_token_here
python code/setup/download_required_assets.py --output_root models --vlm_repo your_org/your_vlm
```

## 4. 必须检查的关键文件

BAGEL 目录必须至少包含：
- `llm_config.json`
- `vit_config.json`
- `ae.safetensors`
- `ema.safetensors`

CLIP 目录必须至少包含：
- `config.json`
- `preprocessor_config.json`

VLM 目录必须至少包含：
- `config.json`

如果显存不足，可退回：
- `Qwen/Qwen3.5-9B`

## 5. 其他需要准备的部分

### 5.1 Benchmark 数据
- 路径：`benchmark/style_benchmark_40x10/`
- 需要满足：40 类 x 10 图（总 400）

### 5.1.1 风格 Prompt 文件
- 路径：`resources/styles/style_prompts_40_v1.json`
- 用法：按 `style_id` 读取对应 prompt；缺失才回退到 `in {style_id} style`。

### 5.2 Pair Benchmark 构建
- 产物路径：
  - `results/pair_benchmark_build/preference_pairs.json`
  - `results/pair_benchmark_build/pair_manifest.json`
  - `results/pair_benchmark_build/pair_quality_report.txt`

### 5.3 训练入口
- 目标脚本：`code/train/dpo_training.py`

### 5.4 评估
- 主协议：VLM 主评估 + CLIP 辅助
- 参考：`BASELINES_20260328.md`

## 6. 执行顺序（必须）

1. 下载并验证模型（本文件第 3-4 节）
2. 校验 benchmark（40x10）
3. 构建并校验 pair benchmark（1:3 主配比）
4. 运行 dpo_training.py（先冒烟再小规模）
5. 运行评估并导出结果

## 7. 常见问题

1. 下载失败/超时：重试并启用 `HF_TOKEN`。
2. 磁盘不足：优先保证 BAGEL + VLM 空间。
3. 推理报错：先检查 CUDA、torch 版本、模型路径完整性。
4. VLM 不可用：不要静默降级，先修复再进入 pair 构建。
