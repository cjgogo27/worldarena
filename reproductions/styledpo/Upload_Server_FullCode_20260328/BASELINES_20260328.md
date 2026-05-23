# Style-DPO Baseline 清单（2026-03-28）

本清单按三层组织：评分器层、模型层、消融层。
目标是统一后续实验记录口径，确保每个结果都可复现、可对比。

## 1. 评分器层 Baseline

### 1.1 CLIP Scorer Baseline（早期）
- 用途：快速自动打分与粗筛。
- 数据输入：content 图 + style 图 + generated 图。
- 输出：style/content 对齐分数及排序结果。
- 代码位置：
  - code/data/data_pairing.py（评分/配对主流程）
  - code/repos/bagel-main/eval/**（官方评估脚本参考）
- 当前状态：已跑通过，作为快速 baseline 保留。

### 1.2 VLM Scorer Baseline（主评估）
- 用途：主评分器，用于构造 chosen/rejected 和最终评估。
- 数据输入：同上，强调风格一致性 + 内容保持。
- 输出：更稳定的偏好排序（替代仅 CLIP 的单一评分）。
- 代码位置：
  - code/data/data_pairing.py（VLM 分支）
  - code/evaluate/vlm_eval.py（评估层入口）
- 当前状态：主路线（优先于 CLIP）。

### 1.3 Hybrid Scorer Baseline（建议保留）
- 用途：CLIP 先粗筛，VLM 再精排，平衡速度与质量。
- 目的：降低全量 VLM 成本，保证大规模构对效率。
- 状态：建议项，便于后续大规模数据构建。

## 2. 模型层 Baseline

### 2.1 BAGEL Base（无 DPO）
- 定义：仅使用 BAGEL 原生生成能力，不进行偏好优化。
- 用途：作为所有改进方法的零点对照。
- 代码位置：
  - code/repos/bagel-main/train/pretrain_unified_navit.py
  - code/repos/bagel-main/modeling/bagel/bagel.py

### 2.2 BAGEL + LoRA（无 DPO）
- 定义：只做参数高效微调，不加偏好损失。
- 用途：验证 LoRA 本身收益，和 DPO 解耦。
- 代码位置：
  - code/train/lora_reference.py
  - code/train/lora_utils.py

### 2.3 BAGEL + DPO（无 LoRA）
- 定义：全量/半冻结参数下仅引入 DPO。
- 用途：验证 DPO 单独贡献。
- 关键依赖：
  - code/train/qwen2_dpo_reference.py（公式与实现参考）

### 2.4 BAGEL + LoRA + DPO（目标主模型）
- 定义：当前真实项目目标配置。
- 用途：在显存可控前提下提升风格偏好对齐。
- 当前状态：框架与依赖齐备，主训练入口脚本待放入（见缺失核对单）。

## 3. 消融层 Baseline（Ablation）

### 3.1 评分器消融
- A1: CLIP-only
- A2: VLM-only
- A3: CLIP->VLM 两阶段
- 观察项：偏好对质量、训练收敛速度、最终人工主观偏好一致性。

### 3.2 损失构造消融
- B1: token-level 直接全局平均（历史错误基线）
- B2: per-sample token 聚合（当前正确方法）
- B3: per-sample + 不同 beta
- 观察项：DPO 稳定性、风格强化是否伴随内容破坏。

### 3.3 训练策略消融
- C1: Freeze-heavy（仅 7 个投影层）
- C2: Freeze-light（更多层可训练）
- C3: LoRA rank {4,8,16}
- 观察项：显存、速度、最终质量。

### 3.4 推理参数消融
- D1: cfg_text_scale 变化
- D2: cfg_img_scale 变化
- D3: num_timesteps / timestep_shift 变化
- 观察项：风格强度、内容保持、伪影率。

## 4. 结果记录模板（建议统一）
- 实验 ID
- Baseline 层级（Scorer/Model/Ablation）
- 数据版本（含 benchmark 子集版本号）
- 超参（beta/lr/batch/lora_rank/freeze）
- 指标（CLIP, VLM, 人工偏好）
- 结论（是否进入下一轮）
