# Claude End-to-End Prompt (Qwen/Qwen3.5-9B)

请你在服务器中从头到尾完成 Style-DPO 任务。不要做无关扩展，按以下顺序执行，目标是尽快产出可训练、可评测结果。

## A. 执行目标
1. 准备环境与模型资源。
2. 构建并校验 style benchmark（40x10）。
3. 构建并校验 preference-pair benchmark（1:3 主配比）。
4. 完成并跑通 `code/train/dpo_training.py`。
5. 进行小规模训练并输出 CLIP/VLM 评估结果。

## B. 必须使用的模型与路径
- BAGEL: `ByteDance-Seed/BAGEL-7B-MoT`
- CLIP: `openai/clip-vit-large-patch14`
- VLM: `Qwen/Qwen3.5-9B`
- 项目根目录：`Upload_Server_FullCode_20260328/`
- 模型下载根目录：`models/`

## B.1 数据构造输入（必须按此执行）

### 内容图片（content images）
- 主来源：OmniStyle-150K 的内容图子集。
- 建议在服务器建立软链接到：
  - `resources/datasets/omnistyle/content/`
- 若无软链接，允许直接使用绝对路径（例如 `/data/.../OmniStyle-150K/content/`），但必须在 `pair_manifest.json` 记录实际路径。

### 风格参考与风格标签
- 固定风格基准：`benchmark/style_benchmark_40x10/`
- 其中每个子目录名就是 `style_id`，目录内 10 张图是该风格参考。

### 风格 prompt（构造 pair 时）
- 必须使用风格 prompt 文件：`resources/styles/style_prompts_40_v1.json`。
- 取值规则：优先读取 `prompts[style_id]`。
- 若某个 `style_id` 未命中，才允许回退到：`in {style_id} style`，并在 `pair_quality_report.txt` 记录回退次数。

### 条件组定义（用于 1:3 构对）
- `group_key = (content_id, style_id, prompt_template_id)`
- 同一组内仅随机种子不同。

## C. 先执行（环境与下载）
在项目根目录运行：

```bash
pip install -r requirements.txt
python code/setup/download_required_assets.py \
  --output_root models \
  --bagel_repo ByteDance-Seed/BAGEL-7B-MoT \
  --clip_repo openai/clip-vit-large-patch14 \
  --vlm_repo Qwen/Qwen3.5-9B
```

然后做最小验证：
- `torch.cuda.is_available()` 为 True
- BAGEL 模型目录存在且关键文件齐全
- VLM/CLIP 各完成 1 次成功推理

## D. Benchmark 构建与门禁
### D1. style benchmark
- 路径：`benchmark/style_benchmark_40x10/`
- 必须满足：40 类 x 10 图（总 400）
- 输出：
  - `results/benchmark_build/benchmark_manifest.json`
  - `results/benchmark_build/benchmark_check_report.txt`

### D2. preference-pair benchmark（核心）
- 分组键：`(content_id, style_id, prompt_template_id)`
- 评分：VLM 主评分（Qwen/Qwen3.5-9B），CLIP 辅助
- 主配比：1 positive : 3 negatives
- 规则：组内最高分 chosen，最低 3 个为 rejected
- 约束：`chosen_score > rejected_score` 且 `chosen_score - rejected_score >= delta_min`（默认 0.02）
- 回退：候选不足可 1:2 或 1:1，但必须记录
- 必须先 split 再构对（按 content_id），禁止 train/val/test 泄漏
- 展开权重：每组 m 个负样本时，每条 pair 权重 `1/m`
- 输出：
  - `results/pair_benchmark_build/preference_pairs.json`
  - `results/pair_benchmark_build/pair_manifest.json`
  - `results/pair_benchmark_build/pair_quality_report.txt`
  - `results/pair_benchmark_build/pair_build_failures.jsonl`

## E. 训练与评估
1. 完成 `code/train/dpo_training.py`（基于 chosen/rejected 的 DPO）。
2. 冒烟训练 1-5 step，确认可前向、可反向、无致命错误。
3. 小规模训练并保存 checkpoint/log。
4. 评估输出 CLIP + VLM 指标。

## F. 最小交付物
1. `code/train/dpo_training.py`
2. `results/pair_benchmark_build/preference_pairs.json`
3. 可直接复现的训练命令
4. 小规模训练指标（CLIP + VLM）

## G. 汇报要求（精简）
只在 4 个里程碑汇报：
1. 资源下载与环境验证完成
2. 两类 benchmark 构建完成
3. 冒烟训练完成
4. 小规模训练与评估完成

每次汇报控制在 10 行内，优先给结果与下一步。