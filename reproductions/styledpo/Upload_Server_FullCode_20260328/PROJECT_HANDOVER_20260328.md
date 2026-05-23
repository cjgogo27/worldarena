# Style-DPO 交接总览（2026-03-28）

本文档用于让新加入成员快速理解：目前做了什么、代码在哪里、下一步做什么。

## 1. 已完成内容

### 1.1 代码包整理
- 上传包目录：Upload_Server_FullCode_20260328/
- 已包含：训练参考、数据配对、评估脚本、BAGEL 主仓代码镜像。

### 1.2 偏好对构造链路
- 偏好对数据结构与生成逻辑已具备。
- 关键位置：code/data/data_pairing.py

### 1.3 评估链路
- CLIP 路线：可作为快速 baseline。
- VLM 路线：作为主评估与主构对评分器。
- 评估参考：code/evaluate/ 与 code/repos/bagel-main/eval/

### 1.4 训练前置能力
- LoRA 参考代码在包内。
- Qwen2 DPO 参考公式与实现在包内。
- BAGEL 训练骨架与模型 forward 路径已定位。

### 1.5 Benchmark 数据已放入上传包
- 目标基准：40 种风格 x 每种 10 张图。
- 目标位置：benchmark/style_benchmark_40x10/
- 校验结果：40 个子目录，每个目录 10 张图。

## 2. 关键代码位置索引

### 2.1 数据与偏好对
- code/data/data_pairing.py

### 2.2 训练参考
- code/train/qwen2_dpo_reference.py
- code/train/lora_reference.py
- code/train/lora_utils.py

### 2.3 BAGEL 主干（镜像）
- code/repos/bagel-main/modeling/bagel/bagel.py
- code/repos/bagel-main/train/pretrain_unified_navit.py
- code/repos/bagel-main/train/fsdp_utils.py

### 2.4 评估
- code/evaluate/
- code/repos/bagel-main/eval/

## 3. 当前唯一缺口
- 缺失文件：code/train/dpo_training.py
- 作用：DPO-BAGEL 主训练入口。
- 详见：MISSING_FILES_CHECKLIST_20260328.md

## 4. 下一步执行顺序
1. 补入 code/train/dpo_training.py。
2. 用小样本跑通 1-5 step 冒烟测试（双卡）。
3. 用 benchmark/style_benchmark_40x10 产出候选图并打分。
4. 批量构造 chosen/rejected，启动正式 DPO 训练。
5. 用 CLIP + VLM 双评估导出报告。

## 5. 交接注意事项
- 统一用同一份 benchmark 与配置，避免实验口径漂移。
- 每次训练都记录 baseline 层级（评分器/模型/消融）。
- 训练失败优先检查：dtype 一致性、冻结层设置、显存峰值。
