# GeoSkill 补充材料缺口补齐清单（仓库证据版）

> 目的：把当前仓库中“已经能直接补”的补充材料内容和“仍需作者提供”的内容分开。
> 范围：仅基于 `/data/alice/cjtest/NIPS/geoskill` 内代码、配置、实验产物。

## 0. 结论速览

- 可直接补齐（代码/实验证据充足）：
  - 技能 schema 与字段定义、解析规则、检索公式、在线 JSON 约束、演化循环机制、GeoRC 导出与评分流程、失败样本统计与子图文件产物、可复现实验配置与审计文件。
- 部分可补（有骨架但缺论文级展示材料）：
  - 更系统的失败案例图文、子图可视化图版、ablation 解释性对比、方差与显著性。
- 仍需作者提供（仓库无法独立还原）：
  - 论文最终提交版使用的固定种子/重复次数、统计显著性检验设置、API 成本账单、伦理与风险披露正文、最终 baseline 公平性约束声明。

## 1. 缺口逐项对照（可补 vs 待补）

| 缺口 | 当前状态 | 可从仓库直接补充 | 仍需作者提供 | 关键证据 |
|---|---|---|---|---|
| 1) Skill schema / 例子 / 关系图 | 可补（高） | `Skill` 六字段定义（`skill_text/region_hint/confidence/visual_cues/source_game_id/source_round`）；可给恢复技能与融合技能真实样例；可给图结构构建规则（`region_support`、`cue_composition`）和 co-occurrence 导出逻辑。 | 论文版高清图（含具体图像截图 + 人工标注解释）仍需人工排版。 | `src/skill_parser.py` L7, L127, L169; `src/GeoVista/skill_graph_runtime.py` L16, L54, L64, L84; `experiments/full_100_mytokenland/skill_updates/*.json*` |
| 2) Expert→Skill 提取规则/提示 | 可补（高） | `parse_expert_chain`、`parse_candidate_chain`、`estimate_confidence`、`extract_visual_cues` 的规则化实现可直接写进补充材料。 | 若论文声称“人工标注协议/双人一致性”，仓库中无该标注流程文件。 | `src/skill_parser.py` L84, L116, L127, L169 |
| 3) 在线推理 prompt+JSON+检索打分 | 可补（高） | 可明确 JSON 输出 schema；可写检索融合公式：`combined = alpha*semantic_norm + (1-alpha)*bm25_norm`；可写 multimodal 检索 query 权重与 candidate vote。 | 若需公开最终生产 prompt 的版本演化记录（A/B 历史），仓库无专门版本日志。 | `src/baselines.py` L19, L127, L159, L834, L922; `src/skill_library.py` L46, L65 |
| 4) 演化算法（阈值/终止/版本化） | 可补（高） | 可写失败判定（国家不匹配、坐标缺失、距离阈值）；可写 rollout 回合、每轮 recovered/fused 导出、graph 快照导出；可写 search-evolution 开关与参数。 | 若论文要给“理论收敛证明/复杂度上界”，仓库无形式化推导文本。 | `scripts/run_experiment.py` L184, L854-L855, L1024-L1132; `src/skill_optimizer.py` L128, L239; `configs/review_evolution_georc50_mytokenland.yaml` L33-L38 |
| 5) GeoRC faithfulness protocol | 可补（高） | 可给 prediction->GeoRC challenge 文件导出流程，及三种打分模式（`key_points`/`bipartite`/`vlm_judge`）与 GT 链解析入口。 | 若论文主文使用了某个外部私有评审脚本版本，需要作者确认版本哈希。 | `scripts/export_predictions_to_georc.py` L98, L205, L209; `external_baselines/GeoRC/score.py` L37, L293, L341 |
| 6) Baseline 可复现性 | 可补（中） | 可披露 adapter/official strict 双通道；可给 strict 配置字段；可给 run_audit 中 baseline 存在性与本地结果状态。 | EP-BEV/Sample4Geo 在部分 run audit 中显示 `present:false`，最终论文若声称完整 official 复现需作者补证据。 | `src/baselines.py` L71, L2142, L2161, L2307; `configs/official_strict.yaml` L39-L75; `experiments/*/run_audit.json` |
| 7) Ablation（是否有效） | 可补（中） | 有多组 ablation summary；可报告 rollout1/rollout3 与 w/o-skill 结果及生成技能数。 | 当前部分 ablation 全量报错（`num_errors=50`）导致性能对比不可解释，需作者补“有效运行”组。 | `experiments/ablation_geovista_skill_graph_georc50_*/summary_metrics.json` |
| 8) 方差/显著性 | 部分可补 | 可先报告已有多次运行目录与 summary 文件。 | 缺少统一多随机种子重复、显著性检验脚本和 p-value 报告。 | `experiments/**/summary_metrics.json` |
| 9) 失败案例分析 | 可补（中） | 可给失败样本 ID、错误类型（超时/重试失败）、失败比例；可从 `skill_dataset_assets` 提取失败分布。 | 论文级“图-文对照失败图库”（含原图、理由、改进建议）仍需人工整理。 | `experiments/review_evolution_georc50_mytokenland/skill_dataset_assets/skill_conditioned_v3_latest.jsonl` |
| 10) 计算成本/资源 | 部分可补 | 可给 `initial_pass_elapsed_seconds`、`rollout_elapsed_seconds`、`elapsed_seconds`。 | 缺 token 使用量、API 计费、显存/GPU 型号与峰值监控日志。 | `scripts/run_experiment.py` L1003-L1004, L1118-L1137; `experiments/*/summary_metrics.json` |
| 11) 可复现实验工件 | 可补（高） | 有配置、审计、预测、指标、rollout trace、skill updates。 | 若要 artifact DOI / 固定快照包（camera-ready 要求）需作者打包发布。 | `configs/*.yaml`, `experiments/*/run_audit.json`, `experiments/*/*latest*.json` |
| 12) 伦理/风险/泄漏防护 | 缺失 | 可在方法层描述“仅街景图推断，不做人脸识别”等一般性约束（需谨慎）。 | 正式伦理声明、数据许可、潜在滥用风险与缓解策略文本需作者撰写。 | 当前仓库未见专门 ethics 文档 |

## 2. 六个优先缺口的可直接补写内容（可粘贴到 Supplementary）

### 2.1 Skill schema、示例与关系（优先级 P0）

可写要点：

1. 技能原子定义采用结构化 dataclass：
   - `skill_text`（规则文本）
   - `region_hint`（洲级先验）
   - `confidence`（由文本置信触发词估计）
   - `visual_cues`（关键词或回退 token）
   - `source_game_id`、`source_round`（可追溯性）
2. 图关系由两类边构成：
   - `region_support`：region 一致时建立强边（weight=1.0）
   - `cue_composition`：视觉 cue 重叠时建立组合边（weight=0.6）
3. rollout 期间从预测记录汇总 co-occurrence 边，形成可审计的动态图快照。

可引用的真实产物：

- `experiments/full_100_mytokenland/skill_updates/recovered_external_geovista_skill_graph_r1_20260401_052215.jsonl`（12 条）
- `experiments/full_100_mytokenland/skill_updates/fused_external_geovista_skill_graph_r1_20260401_052215.jsonl`（3 条）
- `experiments/full_100_mytokenland/skill_updates/graph_external_geovista_skill_graph_latest.json`（`edge_count=80`）

### 2.2 Expert->Skill 抽取规则与 prompt（优先级 P0）

可写要点：

1. 专家链文本由 `parse_expert_chain` 按 `Round/Reasoning/Conclusion` 解析。
2. 候选链文本由 `parse_candidate_chain` 行级解析；结论行用于推断 region。
3. `estimate_confidence` 使用关键词触发规则（如 `definitely/likely/possibly`）映射到离散置信度区间。
4. `extract_visual_cues` 先匹配预定义 cue 词表，不足时回退到 token 抽取。

### 2.3 在线推理 JSON 约束、检索打分与投票（优先级 P0）

可写要点：

1. 所有方法都被约束输出统一 JSON schema（country/country_code/region/city/province_or_state/address/confidence/reasoning/evidence）。
2. 技能检索采用 BM25 + 语义相似度归一化融合：
   - `combined = alpha * semantic_norm + (1-alpha) * bm25_norm`
3. 在 skill-graph 路径中引入 country shortlist + 多候选投票选择，输出带 `candidate_vote` 的可解释记录。

### 2.4 演化算法细节（阈值、停止条件、版本化）（优先级 P0）

可写要点：

1. 失败定义：
   - 预测报错；或
   - 国家不匹配；或
   - 坐标无效；或
   - 距离误差 > `failure_distance_km`。
2. 每轮 rollout：
   - 选失败样本（或 geovista 全记录）
   - 生成 `recovered` 技能，按配置可 `fuse`
   - 增量加入 `SkillLibrary`
   - 重新跑失败样本
   - 落盘 round 级指标和快照
3. 可追溯文件命名：
   - `recovered_{method}_r{idx}_{timestamp}.jsonl`
   - `fused_{method}_r{idx}_{timestamp}.jsonl`
   - `graph_{method}_r{idx}_{timestamp}.json`

当前可量化示例：

- `review_evolution_georc50_mytokenland`：3 轮共恢复 36 条（12/12/12）
- `ablation_geovista_skill_graph_georc50_rollout3`：3 轮共恢复 31 条（8/8/15）
- `full_100_mytokenland`：恢复 12 + 融合 3（共 15）

### 2.5 GeoRC faithfulness 协议（优先级 P0）

可写要点：

1. 预测可导出为 GeoRC challenge 目录下 `candidate_reasoning_chain_*` 文本与 `candidate_prediction_*` JSON。
2. GeoRC 评分脚本支持三种模式：`key_points`、`bipartite`、`vlm_judge`。
3. GT 链文件优先顺序由 `_resolve_gt_path` 控制，可避免因 GT 文件名差异导致评分失败。

### 2.6 失败案例与子图分析（优先级 P0）

可写要点：

1. 在 `review_evolution_georc50_mytokenland` 的 `skill_conditioned_v3_latest.jsonl` 中：
   - 总样本 50
   - `has_error=6`
   - timeout-like 失败 5（主因：上游 API read timeout）
2. 在 `ablation_geovista_skill_graph_georc50_rollout3`：
   - `num_errors=50`
   - `valid_coordinate_rate=0`
   - latest 图快照 `edge_count=0`，说明此 run 未形成有效图关系。

## 3. 关键不一致与风险（建议在补充材料显式声明）

### 3.1 Acc@k 阈值文档-代码漂移

- `src/evaluator.py` 当前阈值：`[10, 25, 200, 750, 2000]`
- `geo_localization_metrics.py` 默认与 `docs/evaluation_metrics.md` 仍写：`[1, 10, 25, 100, 200, 750, 2500]`

建议：Supplementary 明确“主结果使用 `evaluate_predictions` 的阈值集合”，并附兼容性说明，避免复现实验对不上。

### 3.2 v4 实际执行路径

- `skill_conditioned_v4_predict` 当前是 `stabilized_via_v3` 旁路实现（直接调用 v3），而不是完整 A/B/C 投票流程。
- 若论文正文声称使用完整 v4，需要同步修正文案或给出恢复后的 commit。

## 4. 建议作者补充（仓库无法自动补齐）

以下内容建议你补一页“Author-provided Addendum”后可完全闭环：

1. 统计显著性协议：
   - 每方法种子数、重复次数、检验方法（如 bootstrap / permutation / t-test）
   - 置信区间与 p-value 报告格式
2. 成本账单：
   - token 输入/输出、API 成本、总 GPU/CPU 时长、峰值显存
3. 最终 camera-ready 实验声明：
   - 最终使用的 config 文件、模型版本、快照时间戳
   - external baseline 的“official run 证明”与公平性约束
4. 伦理与风险：
   - 数据许可、隐私处理、地理推断误用风险与缓解策略

## 5. 可直接引用的实验摘要（现有数字）

### 5.1 review_evolution_georc50_mytokenland（skill_conditioned_v3）

- `country_accuracy=0.66`
- `continent_accuracy=0.8367`
- `valid_coordinate_rate=0.88`
- `recovered_failures=11`
- `generated_skill_count=36`
- `rollout_rounds=3`
- `num_errors=6`

来源：`experiments/review_evolution_georc50_mytokenland/summary_metrics.json`

### 5.2 full_100_mytokenland（direct vs geovista_skill_graph）

- direct_vlm：`country_accuracy=0.56`, `continent_accuracy=0.88`
- external_geovista_skill_graph：`country_accuracy=0.58`, `continent_accuracy=0.90`

来源：`experiments/full_100_mytokenland/summary_metrics.json`

### 5.3 ablation_geovista_skill_graph_georc50_rollout3（异常 run）

- `num_errors=50`
- `valid_coordinate_rate=0`
- `generated_skill_count=31`
- `skill_graph_versions=3`
- `remaining_failed_cases=50`

来源：`experiments/ablation_geovista_skill_graph_georc50_rollout3/summary_metrics.json`

---

如果你同意，我下一步可以基于本文件再生成一个“可直接贴到论文 Supplementary 的英文版（含结构化段落模板）”，并把上面每条转成 camera-ready 的叙述语气。
