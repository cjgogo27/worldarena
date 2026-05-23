# Style-DPO v2.1 FullCode 打包说明

## 目标理解（本项目要做什么）

本项目目标不是仅做风格迁移推理，而是：
1. 基于 BAGEL 模型进行风格迁移样本生成。
2. 构建自动化 preference pair（chosen/rejected）数据。
3. 使用 DPO 对 policy model 进行偏好优化（含 LoRA 方案）。
4. 在服务器上进行训练与评估闭环（生成、训练、评估、对比）。

## 本次补包内容

在原 `Upload_Server` 基础上，新增了完整源码仓库快照到：

- `code/repos/bagel-main/`（BAGEL 主干：modeling/train/data/eval/inferencer）
- `code/repos/bagel-eval/`（BAGEL 评估与风格迁移相关代码）
- `code/repos/k-lora-main/`（LoRA 训练参考实现）
- `code/repos/qwen-dpo-main/`（DPO 损失参考实现）
- `code/repos/rb-modulation-main/`（VLM/评估辅助代码）

并保留之前你已有的：
- `code/train/*.py`、`code/data/data_pairing.py`、`code/evaluate/*.py`
- `config.yaml`、`requirements.txt`
- 全部部署文档与检查清单

## 本次明确排除项（避免上传无关大文件）

以下内容有意不打进包：
- `.venv/`、`site-packages/`（本地环境副本，不可移植）
- `*.whl`（本地 flash-attn 轮子，服务器环境可能不兼容）
- `*.pth` 大权重（应在服务器按需下载/挂载）
- `assets/`、`pictures/`、`test_images/`、`checkpoints/` 等非训练必需资源
- `*.ipynb` 大 notebook（可后续按需单独同步）

## 新包信息

- 目录：`Upload_Server_FullCode_20260328`
- 压缩包：`Style-DPO-v2.1-fullcode.zip`
- 规模：394 个文件，约 5.36 MB（未压缩）

## 服务器建议步骤

1. 上传并解压 `Style-DPO-v2.1-fullcode.zip`。
2. 先看 `FINAL_DEPLOYMENT_CHECKLIST.md`。
3. 安装依赖后，优先打通：
   - `code/repos/bagel-main/inferencer.py`（基线推理）
   - `code/data/data_pairing.py`（样本对构建）
   - DPO 训练脚本（由 `code/train/` 与 `code/repos/*` 参考整合）
4. 如需指标评估，再接 `code/repos/bagel-eval/` 中的对应脚本。

## 关键提醒

这份包是“代码完整版”而不是“模型权重包”。
模型权重、数据集和中间结果仍建议在服务器侧按路径统一管理，避免重复传输与环境不一致。
