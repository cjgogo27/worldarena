# GeoSkill 评测指标（GeoLocalizationMetrics + GeoRC）

本文档定义当前仓库的统一评测口径：

1. 距离和阈值精度计算使用 [geo_localization_metrics.py](../geo_localization_metrics.py)
2. 分类指标（国家/大洲）与 GeoRC 数据格式对齐
3. 旧自定义指标（幻觉率、专家链 F1、skill coverage/reuse/depth）不再作为主评测输出

实现入口：

1. [src/evaluator.py](../src/evaluator.py)
2. [scripts/run_experiment.py](../scripts/run_experiment.py)
3. [scripts/run_ablation.py](../scripts/run_ablation.py)
4. [scripts/run_pilot.py](../scripts/run_pilot.py)

## 输入 record 约定

每条样本至少包含：

1. GT
   - `ground_truth_country`
   - `ground_truth_lat`
   - `ground_truth_lng`
2. Prediction
   - `prediction.predicted_country`
   - `prediction.predicted_lat`
   - `prediction.predicted_lng`

## 指标集合（当前主输出）

1. `country_accuracy`
2. `continent_accuracy`
3. `distance_error_km_median`
4. `distance_error_km_mean_valid_only`
5. `distance_error_km_mean_penalized`
6. `distance_error_km_std_penalized`
7. `valid_coordinate_rate`
8. `coverage_rate`
9. `Acc@1km`
10. `Acc@10km`
11. `Acc@25km`
12. `Acc@100km`
13. `Acc@200km`
14. `Acc@750km`
15. `Acc@2500km`

其中阈值集合固定为：`[1, 10, 25, 100, 200, 750, 2500]`。

## 计算公式

设样本总数为 $N$。

1. 国家准确率

$$
\text{country\_accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = c_i]
$$

2. 大洲准确率

$$
\text{continent\_accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[R(\hat{c}_i)=R(c_i)]
$$

其中 $R(\cdot)$ 使用 [src/skill_parser.py](../src/skill_parser.py) 里的国家到大洲映射。

3. 距离误差

有效坐标样本使用 Haversine 距离：

$$
d_i = \text{Haversine}(\hat{\phi}_i,\hat{\lambda}_i,\phi_i,\lambda_i)
$$

无效坐标样本使用惩罚值：

$$
d_i = 20037.0
$$

4. 距离统计

$$
\text{distance\_error\_km\_mean\_penalized} = \frac{1}{N}\sum_{i=1}^{N}d_i
$$

$$
\text{distance\_error\_km\_mean\_valid\_only} = \frac{1}{N_{valid}}\sum_{i\in valid}d_i
$$

`median/std` 分别对惩罚后的距离集合计算中位数和标准差。

5. 覆盖率

$$
\text{valid\_coordinate\_rate} = \frac{N_{valid}}{N}
$$

`coverage_rate` 与该定义一致，用于和 [geo_localization_metrics.py](../geo_localization_metrics.py) 输出字段对齐。

6. 阈值精度

$$
\text{Acc@Tkm} = \frac{\#\{i: d_i\le T\}}{N}
$$

## 不再作为主评测输出的旧自定义指标

以下字段已从 `evaluate_predictions` 主输出移除：

1. `heuristic_hallucination_rate`
2. `expert_chain_token_f1`
3. `skill_coverage`
4. `skill_reuse_rate`
5. `skill_composition_depth`

说明：如果需要误差分析，可在专门分析脚本里单独计算，不作为主实验指标。

## 与新 Pipeline 的关系

当前 pipeline 约定：

1. Main agent（小模型）负责基于 skill library 推理
2. LLM（大模型）负责基于 rollout 的成功/失败样本更新 skill
3. 评测层只关心最终预测记录，和模型大小无关

因此，评测代码保持“输入预测记录 -> 输出统一指标”的稳定接口，不绑定具体推理链路。
