# 缺失文件核对单（2026-03-28）

结论：当前上传包只缺 1 个关键文件。

## 缺失项
- 缺失文件：code/train/dpo_training.py
- 角色：DPO-BAGEL 主训练入口（policy/ref + chosen/rejected + DPO loss + 训练循环）
- 备注：其余依赖文件已在包内。

## 已在包内的相关依赖（已核对）
- code/train/qwen2_dpo_reference.py
- code/train/lora_reference.py
- code/train/lora_utils.py
- code/train/batch_reference.py
- code/data/data_pairing.py
- code/repos/bagel-main/modeling/bagel/bagel.py
- code/repos/bagel-main/train/pretrain_unified_navit.py

## 建议你补入后的最终目标路径
- code/train/dpo_training.py

## 补入后最小校验
1. python -m py_compile code/train/dpo_training.py
2. torchrun --nproc_per_node=2 code/train/dpo_training.py --help
3. 检查 config.yaml 中 dpo/lora 参数是否被脚本读取
