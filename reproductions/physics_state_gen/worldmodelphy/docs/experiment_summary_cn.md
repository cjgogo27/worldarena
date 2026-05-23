# WorldModelPhy 实验总报告（中文汇总版）

> 本文档汇总当前 `worldmodelphy` 项目中**所有已完成实验**、可用的**结果文件 / 图片 / GIF / 表格**，以及从这些实验中得到的核心 insight。

项目根目录：`/data/alice/cjtest/worldmodelphy`  
总结果文件：`/data/alice/cjtest/worldmodelphy/artifacts/summary.json`  
总指标表：`/data/alice/cjtest/worldmodelphy/docs/assets/tables/table_quantitative_results.csv`

---

## 1. 这份文档包含什么

1. 已进行的全部实验清单
2. 每个实验的核心定量结果
3. 可直接查看的图与 GIF 路径
4. 对“世界模型内部是否涌现了物理”的综合判断
5. 对 `From Kepler to Newton`、当前 Video Generative Model、以及图像输入小模型实验的统一回答

---

## 2. 已完成实验总览

### 2.1 数据任务

已完成 5 个运动族：

- `circular`
- `projectile`
- `bounce`
- `pendulum`
- `two_body`

### 2.2 模型族

已完成 3 个模型族：

- `gru`：CNN encoder → GRU → CNN decoder
- `local`：CNN encoder → local temporal attention → CNN decoder
- `bottleneck`：CNN encoder → explicit state bottleneck → temporal transition → CNN decoder

### 2.3 上下文设置

- `short`：4 帧
- `long`：16 帧

### 2.4 已跑完的实验名

共 **16** 组：

1. `circular_gru_short`
2. `circular_gru_long`
3. `projectile_gru_short`
4. `projectile_gru_long`
5. `bounce_gru_short`
6. `bounce_gru_long`
7. `pendulum_gru_short`
8. `pendulum_gru_long`
9. `two_body_gru_short`
10. `two_body_gru_long`
11. `circular_local_short`
12. `projectile_local_short`
13. `pendulum_local_short`
14. `circular_bottleneck_short`
15. `projectile_bottleneck_short`
16. `pendulum_bottleneck_short`

---

## 3. 统一实验设置

### 3.1 数据规格

- 分辨率：64×64
- 帧长：32
- 灰度、纯背景
- train / val / test / OOD = 96 / 24 / 24 / 24

### 3.2 评测指标

- `frame_mse`：像素重建 / rollout 误差
- `trajectory_mse`：轨迹误差
- `velocity_mse`：速度误差
- `acceleration_mse`：加速度误差
- `probe_r2_mean`：ID 线性 probe 平均 R²
- `ood_probe_r2_mean`：OOD 线性 probe 平均 R²

### 3.3 OOD 含义

每个任务都设置了参数偏移：

- circular：更大半径 / 更高角速度 / 圆心偏移
- projectile：初速度 / 重力变化
- bounce：速度 / restitution / gravity 变化
- pendulum：摆长 / 重力 / 初始角度变化
- two_body：半径 / 角速度 / 质量比变化

---

## 4. 核心定量结果（全部实验）

> 下表只列最关键的 4 个指标：`ID trajectory MSE`、`OOD trajectory MSE`、`ID probe mean R²`、`OOD probe mean R²`。  
> 全部原始指标见：`docs/assets/tables/table_quantitative_results.csv`

| 实验 | ID traj MSE | OOD traj MSE | ID probe R² | OOD probe R² |
|---|---:|---:|---:|---:|
| circular_gru_short | 83.53 | 744.08 | 0.05 | -0.02 |
| circular_gru_long | 3.24 | 390.34 | 1.00 | 0.02 |
| projectile_gru_short | 375.47 | 492.01 | 0.75 | -11.66 |
| projectile_gru_long | 5.54 | NaN | 1.00 | -24.91 |
| bounce_gru_short | 71.36 | 644.01 | 1.00 | -0.00 |
| bounce_gru_long | 35.69 | 1303.67 | 1.00 | 0.03 |
| pendulum_gru_short | 57.24 | 905.93 | 0.44 | -0.05 |
| pendulum_gru_long | 1.97 | 703.66 | 1.00 | -0.05 |
| two_body_gru_short | 26.75 | 383.70 | 1.00 | 0.08 |
| two_body_gru_long | 0.33 | 208.00 | 1.00 | 0.16 |
| circular_local_short | 311.21 | 945.99 | 1.00 | 0.07 |
| projectile_local_short | 1964.77 | 898.36 | 1.00 | -37.62 |
| pendulum_local_short | 182.21 | 948.53 | 0.99 | -1647289.00 |
| circular_bottleneck_short | 230.12 | 1150.23 | 0.95 | -0.02 |
| projectile_bottleneck_short | 374.57 | 506.44 | 0.77 | -13.51 |
| pendulum_bottleneck_short | 182.09 | 973.19 | 0.97 | -0.05 |

---

## 5. 分任务解读

## 5.1 Circular

### 结果

- `gru_short`：ID 83.53 → OOD 744.08
- `gru_long`：ID 3.24 → OOD 390.34
- `local_short`：ID 311.21 → OOD 945.99
- `bottleneck_short`：ID 230.12 → OOD 1150.23

### insight

- **长上下文 GRU** 在 ID 上最强
- 但 OOD 仍然很差
- `local attention` 与 `bottleneck` 都没有比 baseline 更物理

### 判断

这组结果最像：

> long-context GRU 学会了非常强的轨迹模板拟合，但没有得到稳健的 OOD 动力学。

---

## 5.2 Projectile

### 结果

- `gru_short`：ID 375.47 → OOD 492.01
- `gru_long`：ID 5.54 → OOD NaN
- `local_short`：ID 1964.77 → OOD 898.36
- `bottleneck_short`：ID 374.57 → OOD 506.44

### insight

- `gru_long` 在 ID 上非常强
- 但 OOD 直接崩掉，出现 `NaN`
- 说明“分布内越好”并不等于“物理越强”

### 判断

抛体运动是本项目里最典型的一组证据：

> 可以把训练分布内轨迹学得非常像，但只要重力 / 初速发生偏移，模型就失去稳定 rollout 能力。

---

## 5.3 Bounce

### 结果

- `gru_short`：ID 71.36 → OOD 644.01
- `gru_long`：ID 35.69 → OOD 1303.67

### insight

- 长上下文在 ID 上更好
- 但 OOD 明显更差
- bounce 强化了一个重要结论：**碰撞这种不光滑动力学特别容易暴露模板学习**

### 判断

> 模型学会了常见 bounce pattern，但没有学会更稳定的边界 / restitution 动力学。

---

## 5.4 Pendulum

### 结果

- `gru_short`：ID 57.24 → OOD 905.93
- `gru_long`：ID 1.97 → OOD 703.66
- `local_short`：ID 182.21 → OOD 948.53
- `bottleneck_short`：ID 182.09 → OOD 973.19

### insight

- pendulum 是最能验证“周期系统是否只是被记住了形状”的任务
- `gru_long` 在 ID 上接近完美
- 但 OOD 仍然大幅退化

### 判断

> 周期结构很适合被背成模板，因此 pendulum 特别支持“Keplerian trajectory fitting 而不是 Newtonian dynamics”的解释。

---

## 5.5 Two-body

### 结果

- `gru_short`：ID 26.75 → OOD 383.70
- `gru_long`：ID 0.33 → OOD 208.00

### insight

- 这是当前扩展实验里表面上最“成功”的一组
- 说明模型对**整体结构**可以跟踪得更稳
- 但当前评测使用的是简化中心量 / 平均量，不足以证明它学到了对象级守恒律

### 判断

> 可以说它学会了多体场景里的稳定视觉结构，但还不能说学会了严格意义上的 two-body physics。

---

## 6. 分模型解读

## 6.1 GRU baseline

### 总体结论

- **long context** 基本都让 ID 更强
- 但 OOD 没有同步变强，甚至经常更差

### insight

这与 `From Kepler to Newton` 的启发一致：

> 长上下文给了模型更多“背整段轨迹”的空间。

---

## 6.2 Local-attention baseline

### 总体结论

- 没有自动比 GRU 更“物理”
- 在 circular / projectile / pendulum 上都明显不优

### 代表性结果

- `circular_local_short`：ID 311.21
- `projectile_local_short`：ID 1964.77
- `pendulum_local_short`：OOD probe 极端崩坏

### insight

> “局部 attention” 本身不是足够强的物理归纳偏置。

---

## 6.3 Explicit state bottleneck

### 总体结论

- 压缩 state 并没有自动提高 OOD physics
- 只靠“瓶颈小”不足以逼出真正的动力学抽象

### 代表性结果

- `circular_bottleneck_short`：ID 230.12，OOD 1150.23
- `projectile_bottleneck_short`：ID 374.57，OOD 506.44
- `pendulum_bottleneck_short`：ID 182.09，OOD 973.19

### insight

> bottleneck 需要配合更强的 dynamics loss / intervention setup，单独使用不够。

---

## 7. 最重要的 insight

## 7.1 视觉真实性 ≠ 物理理解

多个实验都出现：

- `frame_mse` 不一定很差
- 但 `trajectory_mse` / `OOD probe R²` 明显崩掉

这说明：

> 像素层面“看起来还行”，不代表内部真的学到了物理规律。

---

## 7.2 长上下文更像“模板记忆增强器”

最清楚的例子：

- `circular_gru_long`
- `pendulum_gru_long`
- `projectile_gru_long`

共同模式：

- ID 非常强
- OOD 没有对应提升

这最接近：

> **学会了训练流形上的轨迹模板，而不是可迁移的局部物理规律。**

---

## 7.3 ID 线性 probe 会系统性高估“物理涌现”

典型例子：

- `circular_gru_long`：ID probe 1.00，OOD 0.02
- `pendulum_gru_long`：ID probe 1.00，OOD -0.05
- `projectile_gru_long`：ID probe 1.00，OOD -24.91

因此：

> 如果只看 ID probe，很容易误判“模型学到了物理”。

必须同时要求：

- ID 好
- OOD 也好
- rollout 也稳

---

## 7.4 新基线没有推翻原结论

无论是：

- `local attention`
- `explicit bottleneck`

都没有展示出明确更强的 OOD 物理泛化。

这意味着：

> 问题不只是“GRU 不够好”，而更像是：当前像素预测目标本身就鼓励了模板学习。

---

## 8. 对“世界模型内部是否涌现了物理？”的最终回答

### 我的结论

- **弱涌现：有**
- **强涌现：证据不足**

更准确地说：

> 模型学到了 motion statistics、trajectory template、以及 physics-flavored latent representations；  
> 但没有展示出稳健、可迁移、可外推的 physics world model。

---

## 9. 对用户三个问题的统一回答

## 9.1 问题 1：`From Kepler to Newton`

这篇论文最重要的结论是：

> 归纳偏置，尤其是**时间局部性**，会影响模型学到的是“全局轨迹形状”还是“局部动力学规律”。

本项目的结果支持这个视角：

- long context 更像轨迹模板拟合
- short context 不一定最好，但至少少了背整段曲线的自由度

---

## 9.2 问题 2：现在 Video Generative Model 中是否涌现了物理？

本项目结论与现有文献一致：

> 没有强证据表明已经稳定涌现了可泛化的物理规律。

更像：

- 会模仿很多视频里的动力学表象
- 会形成局部“物理感”
- 但在 OOD / 守恒 / 因果泛化上依然薄弱

---

## 9.3 问题 3(a)：模型可以多小？

从这批实验看：

- **sub-1M** 级模型已经足够把训练分布拟合得很好
- 但“拟合得好”不等于“学会物理”

因此：

> 模型可以很小，但小模型同样会走向轨迹模板学习。

---

## 9.4 问题 3(b)：模型表现怎么样？有哪些 failure mode？

主要 failure mode：

1. **高 ID、低 OOD**
2. **碰撞类任务更容易暴露模板学习**
3. **周期任务更容易被背成形状**
4. **probe 在 ID 上过度乐观**
5. **多物体整体可跟踪，不等于学到对象级守恒**

---

## 9.5 问题 3(c)：怎么测量模型学没学到物理规律？

推荐至少三层：

1. **Rollout 层**：trajectory / velocity / acceleration
2. **Representation 层**：ID + OOD probe
3. **Intervention 层**：改重力、初速度、restitution、质量比，看 latent 是否有一致响应

如果没有第 2、3 层，只看观感或像素误差，很容易误判。

---

## 10. 图、GIF 与结果文件索引

## 10.1 汇总文件

- 总结果 JSON：`/data/alice/cjtest/worldmodelphy/artifacts/summary.json`
- 总指标 CSV：`/data/alice/cjtest/worldmodelphy/docs/assets/tables/table_quantitative_results.csv`

## 10.2 汇总图

- OOD 对比图：`/data/alice/cjtest/worldmodelphy/docs/assets/figures/fig_ood_failures.png`
- Probe 图：`/data/alice/cjtest/worldmodelphy/docs/assets/figures/fig_probe_accuracy.png`
- ID 对比图：`/data/alice/cjtest/worldmodelphy/docs/assets/figures/fig_id_generation_comparison.png`

## 10.3 代表性 loss 曲线

- `fig_loss_curve_circular_gru_long.png`
- `fig_loss_curve_projectile_gru_long.png`
- `fig_loss_curve_bounce_gru_long.png`
- `fig_loss_curve_pendulum_gru_long.png`
- `fig_loss_curve_two_body_gru_long.png`
- `fig_loss_curve_circular_local_short.png`
- `fig_loss_curve_circular_bottleneck_short.png`

完整目录：`/data/alice/cjtest/worldmodelphy/docs/assets/figures/`

## 10.4 GIF 索引

### Circular
- `gif_test_circular_gru_short.gif`
- `gif_test_circular_gru_long.gif`
- `gif_ood_circular_gru_short.gif`
- `gif_ood_circular_gru_long.gif`
- `gif_test_circular_local_short.gif`
- `gif_ood_circular_local_short.gif`
- `gif_test_circular_bottleneck_short.gif`
- `gif_ood_circular_bottleneck_short.gif`

### Projectile
- `gif_test_projectile_gru_short.gif`
- `gif_test_projectile_gru_long.gif`
- `gif_ood_projectile_gru_short.gif`
- `gif_ood_projectile_gru_long.gif`
- `gif_test_projectile_local_short.gif`
- `gif_ood_projectile_local_short.gif`
- `gif_test_projectile_bottleneck_short.gif`
- `gif_ood_projectile_bottleneck_short.gif`

### Bounce
- `gif_test_bounce_gru_short.gif`
- `gif_test_bounce_gru_long.gif`
- `gif_ood_bounce_gru_short.gif`
- `gif_ood_bounce_gru_long.gif`

### Pendulum
- `gif_test_pendulum_gru_short.gif`
- `gif_test_pendulum_gru_long.gif`
- `gif_ood_pendulum_gru_short.gif`
- `gif_ood_pendulum_gru_long.gif`
- `gif_test_pendulum_local_short.gif`
- `gif_ood_pendulum_local_short.gif`
- `gif_test_pendulum_bottleneck_short.gif`
- `gif_ood_pendulum_bottleneck_short.gif`

### Two-body
- `gif_test_two_body_gru_short.gif`
- `gif_test_two_body_gru_long.gif`
- `gif_ood_two_body_gru_short.gif`
- `gif_ood_two_body_gru_long.gif`

完整目录：`/data/alice/cjtest/worldmodelphy/docs/assets/videos/`

---

## 11. 结论（一句话版本）

> 这套实验最支持的结论不是“视频世界模型已经学会了物理”，而是：  
> **它们已经能学到很多像物理的表征与轨迹模板，但距离稳健、可外推、可迁移的 physics world model 还有明显差距。**
