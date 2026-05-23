# Claude 服务器执行 Prompt（Style-DPO，精准版）

你是一个资深的多模态训练工程师。请在 Linux 服务器上继续完成本项目，目标是让 BAGEL + DPO 训练链路可稳定运行并产出可评估结果。

## -1. 首次启动必须先做（环境与资源准备）
在开始 Benchmark/Pair 构建和训练前，必须先完成以下步骤并验证：

1. 创建并激活 Python 环境（建议 3.10）
2. 安装依赖：
  - `pip install -r requirements.txt`
  - 如需：`pip install flash_attn --no-build-isolation`
3. 运行下载脚本（推荐）：`python code/setup/download_required_assets.py --output_root models --bagel_repo ByteDance-Seed/BAGEL-7B-MoT --clip_repo openai/clip-vit-large-patch14 --vlm_repo Qwen/Qwen3.5-9B`
4. 下载 BAGEL 模型到固定路径（例如 `models/BAGEL-7B-MoT/`）
5. 准备评分器模型：
  - VLM 主评分模型（必须）
  - CLIP 辅助评分模型（建议）
6. 验证 GPU/显存与 CUDA 可用（`nvidia-smi` + Python torch 检查）
7. 做最小加载测试：
  - BAGEL 可加载
  - VLM scorer 可推理 1 个样本
  - CLIP scorer 可推理 1 个样本

补充说明文档：`DOWNLOAD_AND_MODEL_LAYOUT_20260328.md`

若以上任一步失败，不得进入后续构建和训练。

### -1.1 推荐下载命令（可按服务器环境调整）

#### 下载 BAGEL（Hugging Face）
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
save_dir = "models/BAGEL-7B-MoT"
snapshot_download(
   repo_id="ByteDance-Seed/BAGEL-7B-MoT",
   local_dir=save_dir,
   local_dir_use_symlinks=False,
   resume_download=True,
   allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
print("done:", save_dir)
PY
```

#### 下载 CLIP（辅助）
```bash
python - <<'PY'
from transformers import CLIPModel, CLIPProcessor
name = "openai/clip-vit-large-patch14"
CLIPModel.from_pretrained(name)
CLIPProcessor.from_pretrained(name)
print("done:", name)
PY
```

#### VLM 模型
- 优先使用你服务器已有可用的 VLM（与 `code/data/data_pairing.py` 对齐）。
- 若需下载，必须把模型 ID、版本、路径写入 `pair_manifest.json`。

### -1.2 启动前最小验收
- `python -c "import torch; print(torch.cuda.is_available())"` 输出 True
- BAGEL 模型路径存在且关键文件齐全（config + 权重）
- VLM/CLIP 各完成 1 次成功推理

## 0. 工作模式（必须遵守）
- 先审查再动手：先阅读目录和关键文件，再给出执行计划。
- 小步迭代：每完成一个阶段，必须汇报结果、风险、下一步。
- 出错即定位：遇到错误时，不要停在报错；必须给出根因分析、修复方案、修复后验证。
- 不做破坏性操作：禁止 `git reset --hard`、禁止删除关键数据目录。
- 所有命令和修改都要可复现，日志路径要明确。

## 1. 项目根目录与关键现状
项目根目录（上传包）：
- `Upload_Server_FullCode_20260328/`

当前已确认状态：
- 已有 baseline 文档：`BASELINES_20260328.md`
- 已有缺失核对单：`MISSING_FILES_CHECKLIST_20260328.md`
- 已有交接总览：`PROJECT_HANDOVER_20260328.md`
- 已有 benchmark 数据：`benchmark/style_benchmark_40x10/`（40类，每类10图）
- 当前唯一缺失主入口：`code/train/dpo_training.py`
- 正负样本对严格协议：`PAIR_BENCHMARK_PROTOCOL_V1_20260328.md`（必须遵守）

## 2. 本轮核心目标（按顺序完成）
1. 先构建并校验 Benchmark（必须先完成）。
2. 先构建并校验正负样本对 Benchmark（必须在训练前完成）。
3. 补全并落地 `code/train/dpo_training.py`，把 DPO 真正接入 BAGEL 训练流程。
4. 用最小样本做冒烟训练（1-5 steps）验证：能前向、能反向、loss 正常、无 dtype/显存错误。
5. 跑一轮小规模训练并保存 checkpoint 与日志。
6. 用 CLIP + VLM 路线完成小规模评估，输出可读报告。
7. 形成下一轮优化建议（消融优先级 + 参数建议）。

## 2.1 Benchmark 构建（训练前硬性前置）
- 使用固定目录：`benchmark/style_benchmark_40x10/`
- 若目录缺失或不完整，先自动构建并生成校验清单。
- 最低验收标准：
  - 风格类别数 = 40
  - 每类图像数 = 10
  - 总图像数 = 400
- 必须输出以下文件后，才允许进入训练阶段：
  - `results/benchmark_build/benchmark_manifest.json`
  - `results/benchmark_build/benchmark_check_report.txt`

建议清单内容：每个类别名、图片数量、缺失项（如有）、构建时间、来源路径。

## 2.2 正负样本对 Benchmark 构建（训练前硬性前置）
- 必须完全遵守：`PAIR_BENCHMARK_PROTOCOL_V1_20260328.md`。
- 输入来源：`benchmark/style_benchmark_40x10/` 与生成候选图。
- 配对规则：同一内容/风格条件下，按评分器（优先 VLM，CLIP 可辅助）选择 chosen/rejected。
- 风格 prompt 来源：`resources/styles/style_prompts_40_v1.json`（必须优先读取）。
- 仅当某个 `style_id` 在该文件中不存在时，才允许回退 `in {style_id} style`，并统计回退次数。
- **正负样本配比（硬约束）**：每 1 个正样本对应 3 个负样本（`1:3`）。
- 负样本选择规则：在同一条件组中，按评分从低到高选取 Top-3 低分样本作为该正样本的负样本。
- 若某条件组候选不足 4 张导致无法满足 `1:3`，允许临时回退到 `1:1` 或 `1:2`，但必须在质量报告中逐组记录原因和数量。
- 必须保证每条样本满足：
  - chosen 与 rejected 对应同一 prompt / 同一条件组
  - chosen_score > rejected_score
  - 分数字段与来源评分器字段完整
- 输出目录：`results/pair_benchmark_build/`
- 必须输出以下文件后，才允许进入训练阶段：
  - `results/pair_benchmark_build/preference_pairs.json`
  - `results/pair_benchmark_build/pair_manifest.json`
  - `results/pair_benchmark_build/pair_quality_report.txt`
  - `results/pair_benchmark_build/pair_build_failures.jsonl`
- 最低验收标准（可按资源提高）：
  - 有效 pair 数 >= 1000（最低可训练门槛）
  - 无非法 pair（chosen_score <= rejected_score）
  - 正负配比统计清晰：主配比 `1:3` 的占比必须报告
  - 各风格类别都有覆盖统计（避免极端偏科）

建议报告内容：
- pair 总数、各风格 pair 数分布、评分差值分布（均值/分位数）
- 正负配比分布（1:3, 1:2, 1:1 各占比）
- 被过滤样本数量及原因（分差过小、图像损坏、解析失败等）
- 数据版本号与构建命令

### 2.2.1 你必须执行的具体构造算法（不可省略）
1. 按 `(content_id, style_id, prompt_template_id)` 分组，其中 `prompt_template_id` 来自 `resources/styles/style_prompts_40_v1.json`。
2. 每组生成 `K` 张候选图（建议 `K>=6`，最小 `K=4`，使用不同随机种子）。
3. 用 VLM 对组内候选打分并排序（CLIP 仅辅助记录，不作为主排序）。
4. 选最高分为正样本 chosen，选最低 3 张为负样本 rejected_list（形成 `1:3`）。
5. 校验：对每个负样本都满足 `chosen_score > rejected_score`。
6. 校验分差：`chosen_score - rejected_score >= delta_min`（默认 `delta_min=0.02`，可配置）。
6.1 若组内分数几乎相同（`max-min < 1e-6`），直接标记为无信息组并丢弃。
7. 若有效候选不足：
  - 3 张候选 -> 回退 `1:2`
  - 2 张候选 -> 回退 `1:1`
  - 少于 2 张 -> 丢弃该组
8. 把所有回退/丢弃原因写入 `pair_quality_report.txt`。
9. 输出 `preference_pairs.json` 时保留 group_id、seed、scorer 元数据。
10. 训练前把 `1:3` 记录展开成 3 条 DPO pair：`(chosen, rej1)`, `(chosen, rej2)`, `(chosen, rej3)`。
11. 展开权重必须做组归一：若该组有 `m` 个负样本，则每条展开 pair 权重为 `1/m`，该组总权重为 1。
12. 必须先做 split 再构对（按 content_id），禁止 train/val/test 泄漏。
13. 同分排序必须可复现：score 相同用 seed 升序，再用文件名排序。
14. 必须做去重与坏图过滤（至少 pHash 去重 + 文件可读性检查），并写入质量报告。
15. 评分必须可复现：VLM 推理温度设为 0，记录评分模型版本与提示词模板版本。
16. 若 VLM 不可用，不允许静默降级为 CLIP-only；必须标记本轮为无效并输出报告。
17. 若 VLM 与 CLIP 排序分歧率过高，必须先输出诊断报告，再决定是否继续训练。

## 3. DPO 实现要求（技术约束）
- 训练目标：基于 chosen/rejected 的 DPO 损失。
- 参考实现可用：
  - `code/train/qwen2_dpo_reference.py`
  - `code/repos/bagel-main/train/pretrain_unified_navit.py`
  - `code/repos/bagel-main/modeling/bagel/bagel.py`
- 关键要求：
  - 使用 per-sample 的 MSE 聚合（不要做全 token 混合平均导致样本偏置）。
  - 同时计算 policy 与 reference 分支，构建 DPO 对比项。
  - 保持 bfloat16 / float dtype 一致，避免 index put dtype mismatch。
  - 冻结策略和 LoRA 策略可配置化（从 config 读取）。

## 4. 数据与评估要求
- 数据输入：优先复用 `code/data/data_pairing.py` 产出的 chosen/rejected 格式。
- benchmark：`benchmark/style_benchmark_40x10/` 作为固定对照集。
- 训练数据：必须来自 `results/pair_benchmark_build/preference_pairs.json` 或同结构文件。
- 评估输出：
  - CLIP 指标
  - VLM 指标
  - 样例可视化（至少每类 1-2 张）

## 5. 你每一轮必须输出的内容（固定模板）
每完成一轮，请按以下格式返回：

### Round N - 状态
- 完成项：
- 修改文件：
- 运行命令：
- 关键日志路径：
- 指标结果：
- 遇到问题：
- 根因分析：
- 修复动作：
- 下一轮计划：

## 6. 质量门禁（通过后才能进入下一步）
- Benchmark 门禁：40 类 x 10 图校验通过，并生成 manifest/report。
- Pair Benchmark 门禁：`preference_pairs.json` 构建完成且通过质量检查（无非法 pair，数量达标）。
- 泄漏门禁：不得出现跨 split 的 content_id 重叠。
- 权重门禁：展开后 pair 权重配置必须存在并通过抽样检查。
- 稳定性门禁：无信息组占比与丢弃率必须报告；若丢弃率 > 30%，暂停训练并先修数据管线。
- 语法门禁：新增/修改脚本可通过 Python 语法检查。
- 训练门禁：最小训练 step 跑通，无致命报错。
- 产物门禁：checkpoint、日志、评估结果文件均落盘。
- 复现实验门禁：命令可一键重跑。

## 7. 最终交付清单（你要产出）
1. `code/train/dpo_training.py`（主训练入口）
2. `results/pair_benchmark_build/preference_pairs.json`（正负样本对 Benchmark）
3. 一份运行脚本（或命令块）用于快速启动
4. 一份错误修复记录（含关键坑位）
5. 一份结果报告（baseline 对比 + 小规模指标）
6. 下一阶段行动清单（高优先级 3-5 条）

## 8. 立即开始执行
请先做这 4 件事并汇报：
1. 打印项目目录树（关键层级即可）。
2. 先构建并校验 Benchmark，输出 manifest/report。
3. 先构建并校验正负样本对 Benchmark，输出 preference_pairs 与质量报告。
4. 检查并确认 `code/train/dpo_training.py` 是否缺失，给出你计划修改/新增的文件列表；然后实现第一版并进行 1 次最小冒烟测试。

---

建议你在会话第一条就执行（复制即用）：

"先不要训练。请严格按 CLAUDE_SERVER_PROMPT_20260328.md 的 -1 节先完成环境与资源准备：安装依赖、下载 BAGEL、准备 VLM/CLIP、完成最小加载验证。通过后再按顺序做 benchmark -> pair benchmark -> dpo_training.py -> 冒烟训练 -> 小规模训练评估。"
