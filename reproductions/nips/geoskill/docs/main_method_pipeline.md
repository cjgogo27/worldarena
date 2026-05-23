# GeoSkill 主方法实现手册（可直接照着实现）

本文档目标：看完后可以从 0 到 1 复现 GeoSkill 主方法（Skill-Conditioned 系列，重点 v2/v4）的完整执行链。

## 1. 先看结论：主方法的真实执行链

实际执行顺序是：

1. 数据处理（构建样本 + 构建技能库）
2. Method 输入（单张街景图 + 可选技能检索上下文）
3. 中间过程（多阶段推理 / 检索 / 复核）
4. Method 输出（结构化 JSON 预测）
5. Benchmark 评测（country/continent/median+mean distance/valid/Acc@k/token-F1/hallucination）

主入口代码：

- [run_experiment.py main](../scripts/run_experiment.py#L176)
- [方法映射 all_methods](../scripts/run_experiment.py#L233)
- [主方法 v1 映射 skill_conditioned](../scripts/run_experiment.py#L236)
- [主方法 v2/v3/v4 映射](../scripts/run_experiment.py#L237)

## 2. 论文思路到代码的映射

主方法相关论文/思想来源与代码位置：

1. Skill-Conditioned Geolocation（本文方法）
- [skill_conditioned_predict (v1)](../src/baselines.py#L179)
- [skill_conditioned_v2_predict](../src/baselines.py#L357)
- [skill_conditioned_v3_predict](../src/baselines.py#L451)
- [skill_conditioned_v4_predict](../src/baselines.py#L596)

2. GeoCoT（5 步推理）
- [geocot_predict](../src/baselines.py#L268)
- 在 v2/v3/v4 中作为关键子流程（第一阶段）

3. PIGEON 风格层级推理（continent->country->city）
- v4 的 Pass B（函数内部实现）: [skill_conditioned_v4_predict](../src/baselines.py#L596)

4. Self-Consistency（多路推理 + majority vote）
- v4 的 Pass A/B/C 投票逻辑: [skill_conditioned_v4_predict](../src/baselines.py#L596)

5. GRE 风格局部细节指纹
- v3/v4 的 local detail pass: [skill_conditioned_v3_predict](../src/baselines.py#L451), [skill_conditioned_v4_predict](../src/baselines.py#L596)

## 3. 数据集、输入文件、样本构造

### 3.1 用什么数据集

- 数据集：GeoRC（项目落地在 `data/georc`）
- 配置入口： [full.yaml](../configs/full.yaml)
- 代码中默认使用 100 个 game_id，评测 round 1（可参数改）

关键配置：

- 数据根目录 `paths.data_root`: `data/georc`
- 实验输出目录 `paths.experiment_dir`: `experiments/full_100`
- 检索参数：`skills.top_k=5`, `skills.min_score=0.15`

### 3.2 一个样本目录里有什么

示例目录：`data/georc/1NJsXTxIF9GGMDxC/`

- `1NJsXTxIF9GGMDxC_1.png` 到 `_5.png`
- `1NJsXTxIF9GGMDxC_rounds_metadata.json`
- `Human_Expert_1.txt`
- `candidate_reasoning_chain_gpt4_1.txt` 到 `_5.txt`

### 3.3 样本构造函数（method 输入前）

- [make_sample](../scripts/run_experiment.py#L94)

输入：

- `gid`（game id）
- `round_idx`（默认 1）
- 对应图片和 metadata 文件

输出（示例）：

```json
{
  "game_id": "1NJsXTxIF9GGMDxC",
  "round": 1,
  "image_path": ".../1NJsXTxIF9GGMDxC_1.png",
  "ground_truth_country": "kg",
  "ground_truth_lat": 41.40303624959777,
  "ground_truth_lng": 74.02089741449993,
  "expert_chain": "Round 1 ..."
}
```

对应真实 metadata 里的一条 GT（round 1）：

```json
{
  "lat": 41.40303624959777,
  "lng": 74.02089741449993,
  "streakLocationCode": "kg"
}
```

## 4. 阶段 A：离线技能库构建（数据处理核心）

### 4.1 入口

- [build_skill_library in run_experiment](../scripts/run_experiment.py#L69)
- [SkillLibrary](../src/skill_library.py#L11)
- [parse_expert_chain](../src/skill_parser.py#L169)
- [parse_candidate_chain](../src/skill_parser.py#L127)

### 4.2 需要哪些输入数据

每个 game 读取：

1. `Human_Expert_*.txt`
2. `candidate_reasoning_chain_gpt4_1..5.txt`

这一步不训练模型，只做解析与索引。

### 4.3 技能结构（每条 skill 的字段）

定义见 [Skill dataclass](../src/skill_parser.py#L7)：

- `skill_text`: 技能文本
- `region_hint`: 地区提示（按国家映射推断）
- `confidence`: 文本置信度估计
- `visual_cues`: 视觉 cue 关键词
- `source_game_id`
- `source_round`

### 4.4 检索索引与打分

- [add_skills](../src/skill_library.py#L23)
- [retrieve](../src/skill_library.py#L36)

检索是 BM25 + 语义向量混合：

- BM25: `rank_bm25`
- 语义向量: `sentence-transformers/all-MiniLM-L6-v2`
- 融合权重：`alpha`（默认 0.5）

## 5. 阶段 B：在线推理（Method 输入 -> 中间过程 -> 输出）

## 5.1 统一 VLM 调用层

- [VLMClient.query](../src/vlm_client.py#L44)
- [图片预处理 _prepare_image_data_uri](../src/vlm_client.py#L30)
- [JSON 提取 extract_json](../src/vlm_client.py#L80)

输入：

- `image_path`（可空，v4 最后一跳 text-only）
- `system_prompt`
- `user_prompt`
- `temperature`

输出：

- 模型原始文本（后续再解析成结构化预测）

## 5.2 主方法 v2（最稳定、指标最好的一版）

函数： [skill_conditioned_v2_predict](../src/baselines.py#L357)

### v2 流程拆解

1. Stage 1: GeoCoT 结构化推理（图像输入）
- 产出初始国家/地区猜测：`candidate_country`, `candidate_region`

2. Stage 2: Region-targeted 检索
- query 使用 stage1 reasoning + candidate country/region
- 检索 `top_k*4`，`deduplicate_region=False`
- 过滤 `region_hint == candidate_region`
- 偏向更长（更 composed）的 skill

3. Stage 3: Expert skill 复核（图像输入）
- 将 stage1 摘要 + 技能文本一起喂给 VLM
- 让模型确认/修正最终答案

### v2 输入输出样例

输入（函数级）：

```python
skill_conditioned_v2_predict(vlm, skill_library, image_path, top_k=5)
```

输出（结构化预测，示例字段）：

```json
{
  "predicted_country": "th",
  "predicted_region": "asia",
  "predicted_lat": 15.87,
  "predicted_lng": 100.9925,
  "reasoning_text": "...",
  "evidence_spans": ["..."],
  "confidence": 0.75,
  "retrieved_skills": [
    {
      "skill_text": "...",
      "region_hint": "asia",
      "score": 0.88
    }
  ],
  "stage1_prediction": {"predicted_country": "..."},
  "candidate_region": "asia"
}
```

## 5.3 主方法 v4（研究版：多路推理 + 投票 + 自我批判）

函数： [skill_conditioned_v4_predict](../src/baselines.py#L596)

### v4 五大中间阶段

1. Pass A（GeoCoT, T=0.2）
2. Pass B（hierarchical forced-choice, T=0.3）
3. Pass C（adversarial elimination, T=0.4）
4. Local fingerprint（GRE 风格局部特征, T=0.1）
5. 投票 + 坐标平均 + region-targeted 检索 + text-only synthesis（T=0.1）

### v4 阶段 I/O 要点

- A/B/C 输入：同一张图
- A/B/C 输出：各自国家与坐标
- Voting 输入：A/B/C（+ local tie-break）
- Voting 输出：`voted_country`, `voted_region`
- Retrieval 输入：voted 结果 + pass 摘要
- Synthesis 输入：pass 摘要 + local 分析 + 检索技能 + 坐标平均
- Synthesis 输出：最终 JSON 预测

## 6. 阶段 C：统一输出解析与评测

### 6.1 输出 schema 规范化

- [JSON schema instruction](../src/baselines.py#L10)
- [prediction 解析器 _parse_json_prediction](../src/baselines.py#L30)

无论模型吐什么格式，最终都规范到统一字段：

- `predicted_country`
- `predicted_region`
- `predicted_lat`, `predicted_lng`
- `reasoning_text`
- `evidence_spans`
- `confidence`

### 6.2 benchmark 与指标来源

评测入口： [evaluate_predictions](../src/evaluator.py#L81)

指标：

1. Country Accuracy
2. Continent Accuracy（按国家映射到大洲，不直接用模型 region 文本）
3. Distance：median（主指标） + mean_valid_only + mean_penalized（Haversine）
4. Acc@1km / Acc@25km / Acc@150km / Acc@750km / Acc@2500km
5. valid_coordinate_rate
6. heuristic_hallucination_rate
7. expert_chain_token_f1

关键规则（非常重要）：

- 无效坐标会被罚 20037 km，不是丢弃样本
- 因此 distance 和 within 指标分母始终是 N

代码来源： [MAX_PENALTY_KM 与距离策略](../src/evaluator.py#L75)

## 7. 是否需要训练？训练什么部分？

结论：不需要训练。

- 主方法和 baseline 全部是 training-free prompt/inference pipeline
- 没有参数更新，没有反向传播
- “数据处理”仅是技能解析 + 建库 + 检索索引

可训练/可替换的只有工程层部件（非必须）：

1. 替换 embedding model（检索质量会变化）
2. 调整检索融合系数 `alpha`、阈值 `min_score`
3. 调整 prompt 和温度

## 8. 一份可直接复现的实现清单

1. 准备 GeoRC 数据目录结构（每个 game 包含 png、metadata、expert/candidate chain）
2. 实现 `Skill` 解析和 `SkillLibrary.retrieve`（BM25+向量）
3. 实现统一 VLM 调用与 JSON 解析容错
4. 按 v2 或 v4 逐阶段拼装 pipeline
5. 落盘 `latest_predictions.json` 和 `latest_metrics.json`
6. 用统一 evaluator 跑完整指标

运行示例：

```bash
python scripts/run_experiment.py --config configs/full.yaml --methods skill_conditioned_v2
```

## 9. 主方法流程图（Mermaid）

```mermaid
flowchart TD
    A[GeoRC data files\nimage + metadata + expert/candidate chains] --> B[Sample builder\nmake_sample]
    A --> C[Skill extraction\nparse_expert_chain + parse_candidate_chain]
    C --> D[Skill index\nBM25 + sentence-transformer]

    B --> E[Method input\nimage_path]
    D --> F[Retriever]

    E --> G[Pass1 GeoCoT\ncandidate country/region]
    G --> F
    F --> H[Region-targeted skills\nfilter by region_hint]
    H --> I[Pass2 verification\nskill-conditioned prediction]

    I --> J[Unified parser\n_parse_json_prediction]
    J --> K[Record writer\nlatest_predictions.json]
    K --> L[Evaluator\ncountry/region/distance/valid/evidence]
    L --> M[latest_metrics.json + summary_metrics (new metric schema)]
```

## 10. 当前仓库里可直接参考的结果文件

- [full_100 汇总](../experiments/full_100/summary_metrics_v2.json)
- [ablation 汇总](../experiments/ablation/summary_metrics.json)
- [skill_conditioned 预测样例](../experiments/full_100/skill_conditioned/latest_predictions.json)
