# Benchmark 数据说明（40风格 x 10图）

## 1. 目标
- 作为本项目统一的风格基准集。
- 用于：候选生成、评分构对、横向比较。

## 2. 数据位置
- benchmark/style_benchmark_40x10/

## 3. 数据来源
- 来源目录：Infer/style_image_delicate/
- 风格名单来源：Infer/Bagel/accuracy_summary2.csv（40 种风格）

## 4. 校验结果
- 风格类别数：40
- 每类图像数：10
- 总图像数：400
- 目录结构：每个风格一个子目录

## 5. 使用建议
- 不要随意增删类别，保持固定 40 类口径。
- 如需扩展，新增独立版本目录（例如 style_benchmark_50x10_v2）。
- 训练与评估日志中标明 benchmark 版本路径。
