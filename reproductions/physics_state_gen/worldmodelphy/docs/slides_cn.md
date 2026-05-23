# 从开普勒到牛顿：视频生成模型内部是否涌现了物理？

世界模型物理推理实验汇报  
MIT 风格中文组会版本  
项目目录：`/data/alice/cjtest/worldmodelphy`

---

# 目录

1. 问题定义：什么叫“内部涌现了物理”
2. 问题 1：`From Kepler to Newton` 讲了什么
3. 问题 2：现在 Video Generative Model 中是否涌现了物理
4. 问题 3：把输入从坐标换成图片后，能否在小模型中看到物理
5. 扩展实验：bounce / pendulum / two-body / bottleneck / local-attention
6. 结论：我们到底看到了什么，没看到什么

---

# 一、问题定义

## 我们说“模型内部涌现了物理”，至少要满足什么？

不是：

- 视频看起来顺
- 短期轨迹像对的
- 分布内预测误差低

而是：

- 内部表征暴露出 **局部动力学变量**
- 对未见参数有 **分布外泛化**
- failure mode 更像“物理近似误差”，而不是“模板记忆崩溃”

---

# 二、问题 1：`From Kepler to Newton` 讲了什么？

## 核心结论

论文：**From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers**

核心观点：

1. **预测得准，不等于学到了物理**
2. **上下文长度**会显著影响模型到底学成了什么
3. **短上下文 + 局部归纳偏置** 更容易逼模型学局部动力学，而不是整条轨迹模板

---

# `From Kepler to Newton`：它究竟区分了什么？

论文区分的是两种内部表征：

- **Keplerian**：记住全局轨道几何 / 曲线形状
- **Newtonian**：编码局部动力学 / 力 / 速度 / 加速度关系

关键启发：

> 想让世界模型更像物理模型，必须限制它“直接背整段轨迹”的自由度。

这也是我们后面做：

- short vs long context
- local attention
- explicit state bottleneck

---

# 三、问题 2：现在 Video Generative Model 中是否涌现了物理？

## 文献结论

综合：

- **Physics-IQ**
- **Morpheus**
- 近年的 video physics benchmark

更合理的说法不是“已经学会物理”，而是：

> 模型学会了很多 **视觉时间统计规律**，但还没有强证据表明它稳定学会了 **可泛化的因果物理规律**。

---

# 当前 Video Generative Model 的共识画像

## 已有能力

- 短期运动连续性
- 常见场景里的“物理感”
- 很强的视觉 plausibility

## 仍然薄弱

- OOD 参数变化
- 守恒律 / 碰撞 / 支撑关系
- 多物体交互
- 长时 rollout 稳定性
- 可解释内部动力学

结论：

> **弱形式的物理感存在，强形式的 physics world model 证据不足。**

---

# 四、问题 3：把输入从坐标换成图片后，还能学到物理吗？

## 实验思想

把 `From Kepler to Newton` 的思路搬到像素空间：

- 输入不再是 `(x_t, y_t)`
- 输入变成 64×64 灰度视频帧
- 让小模型从像素里自己恢复状态，并继续预测

这比坐标输入更难，因为模型要同时学：

1. **感知**：从像素恢复状态
2. **动力学**：从状态推未来

---

# 我们的基础实验设置

## 数据

- 分辨率：64×64
- 帧长：32
- 灰度、纯背景
- train / val / test / OOD = 96 / 24 / 24 / 24

## 初始两类运动

- 圆周运动
- 抛体运动

## 基础模型

- CNN encoder → GRU → CNN decoder
- short context = 4
- long context = 16

---

# 问题 3(a)：模型可以多小？如果不能更小，为什么？

## 我们的经验结论

从这轮实验看：

- **几十万参数量**已经足够拟合训练分布
- 但这不等于它真的学会了物理

当前模型量级大约在 **sub-1M** 级别，已经能：

- 生成平滑视频
- 在 ID 上得到很低误差

但仍然在 OOD 上系统性失败。

---

# 为什么不能再简单地说“更小也行”？

因为像素输入的下界不是由物理公式复杂度决定，而是由两件事决定：

1. **表征学习容量**：先要看懂小球在哪里、怎么动
2. **时序状态容量**：还要记住速度 / 相位 / 局部动力学

所以“小模型”可以把 **轨迹模板** 学会，
但要把 **可泛化的动力学规律** 学会，难度明显更高。

---

# 问题 3(b)：模型表现怎么样？有哪些 Failure Mode？

## 基础结果：GRU baseline

### Circular

- short: ID traj MSE = **83.53**，OOD = **744.08**
- long: ID traj MSE = **3.24**，OOD = **390.34**

### Projectile

- short: ID traj MSE = **375.47**，OOD = **492.01**
- long: ID traj MSE = **5.54**，OOD = **NaN**（直接崩）

观察：

- **长上下文显著提升 ID**
- **OOD 依然很差，甚至更脆弱**

---

# 扩展实验：为什么要加 bounce / pendulum / two-body？

每个新任务都对应一种更强的“物理检验”：

- **bounce**：离散碰撞 / 非光滑动力学
- **pendulum**：非匀速、重力驱动的连续动力学
- **two-body**：多物体耦合系统

目标不是把 benchmark 做大，
而是看 failure mode 会不会从“模板记忆”转向“真实动力学逼近”。

---

# 扩展结果 1：bounce

## GRU baseline

- short: ID traj MSE = **71.36**，OOD = **644.01**
- long: ID traj MSE = **35.69**，OOD = **1303.67**

解释：

- 长上下文在 ID 上更好
- 但遇到 restitution / speed 的 OOD 偏移后更糟

这很像：

> 模型记住了“常见 bounce 轨迹模板”，但没有学会稳定的碰撞动力学。

---

# 扩展结果 2：pendulum

## GRU baseline

- short: ID traj MSE = **57.24**，OOD = **905.93**
- long: ID traj MSE = **1.97**，OOD = **703.66**

这说明：

- 对周期系统，长上下文特别容易把 **整个周期形状** 拟合得很好
- 但这并不自动转化成 OOD 物理泛化

---

# 扩展结果 3：two-body

## GRU baseline

- short: ID traj MSE = **26.75**，OOD = **383.70**
- long: ID traj MSE = **0.33**，OOD = **208.00**

这是扩展任务里相对“最好看”的一组。

但要小心解释：

- 当前 two-body 评测使用的是简化中心量 / 平均量
- 它说明模型能更稳定地跟踪**整体结构**
- 还不能直接证明模型真的学到了两体守恒关系

---

# 问题 3(b)：Failure Modes 总结

## 我们反复看到 5 类失败

1. **长上下文模板记忆**
   - ID 很强
   - OOD 崩得很快

2. **局部运动抖动**
   - 尤其 short context

3. **碰撞/边界处理失败**
   - bounce 最明显

4. **周期系统外推失败**
   - pendulum 在 OOD 下仍明显退化

5. **多物体结构可跟踪，但守恒律并未被证明学到**

---

# 问题 3(c)：怎么测量模型学没学到物理规律？

## 我们用了三层评估

### 第一层：像素层

- frame MSE

### 第二层：轨迹层

- trajectory MSE
- velocity MSE
- acceleration MSE

### 第三层：表征层

- hidden state linear probe → `vx, vy, ax, ay`

---

# 为什么“只看线性 probe”会误判？

看 Circular / Pendulum 的结果：

- `circular_gru_long`：ID probe mean R² = **1.00**，OOD = **0.02**
- `pendulum_gru_long`：ID probe mean R² = **1.00**，OOD = **-0.05**

这意味着：

> latent state 在训练分布内可以线性暴露动力学量，
> 但这并不意味着它是一个可迁移的 physics representation。

所以判断“学到了物理”，必须要求：

- **ID probe 好**
- **OOD probe 也稳**
- **rollout 也稳**

---

# 新基线 1：local-attention 是否更物理？

## 结果

### Circular

- local-short: ID traj MSE = **311.21**，OOD = **945.99**

### Projectile

- local-short: ID traj MSE = **1964.77**，OOD = **898.36**

### Pendulum

- local-short: ID traj MSE = **182.21**，OOD = **948.53**

结论：

> 仅仅加 local attention，不足以让模型更“物理”；
> 反而在这套小模型里经常更差。

---

# 新基线 2：explicit state bottleneck 是否更物理？

## 结果

### Circular

- bottleneck-short: ID traj MSE = **230.12**，OOD = **1150.23**

### Projectile

- bottleneck-short: ID traj MSE = **374.57**，OOD = **506.44**

### Pendulum

- bottleneck-short: ID traj MSE = **182.09**，OOD = **973.19**

结论：

> 只靠“压小 state”本身，不足以逼出更好的物理泛化。

---

# 一个重要发现：long context 的作用

在 GRU baseline 上，几乎所有任务都呈现：

- **ID 显著提升**
- **OOD 没有同步提升，甚至更差**

这与 `From Kepler to Newton` 的启发非常一致：

> 长上下文更容易让模型走向“全局轨迹模板拟合”；
> 短上下文未必更强，但至少不那么鼓励背整段轨迹。

---

# 回答问题 2：现在 Video Generative Model 中是否涌现出了物理？

## 我的结论

**没有强证据表明已经稳定涌现。**

更准确地说：

- 有 **局部、表面化的物理感**
- 没有看到 **稳健、可外推、可迁移的物理规律表征**

我们的 toy experiment 与文献结论一致：

> 模型更像是在学 **视频动态模板**，而不是学 **因果物理机制**。

---

# 回答问题 3(a)：模型可以多小？

## 本项目的证据

- sub-1M 模型就足以拟合 ID
- 但即使这么小，也能明显“学会模板”
- 所以“能拟合”不是关键，关键是“能否 OOD 保持动力学”

因此：

> 模型可以很小，但小不代表更接近物理；
> 小模型同样会走向模板记忆。

---

# 回答问题 3(b)：模型表现和 failure modes

一句话总结：

> **长上下文 + 像素重建目标** 很容易得到“高 ID、低 OOD”的模板学习器。

failure modes：

- 碰撞处错误
- 周期外推失败
- 参数偏移即崩
- probe 在 ID 上过度乐观
- 多体系统只学到整体形状，不代表学到守恒律

---

# 回答问题 3(c)：如何测量模型学没学到物理规律？

## 推荐标准

1. **Rollout 级**：ID + OOD trajectory / velocity / acceleration
2. **Representation 级**：ID + OOD linear probe
3. **Intervention 级**（未来工作）
   - 改初速度
   - 改重力
   - 改 restitution
   - 看 latent 是否有一致响应

如果没有第 2、3 层，单靠视频观感几乎一定会高估“物理理解”。

---

# 最终结论

## 对“世界模型内部是否涌现了物理？”的回答

### 我的当前回答：

**弱涌现：有**  
**强涌现：证据不足**

更具体地说：

- 模型能学到运动统计规律
- 能在 ID 内形成可解码的动力学相关表征
- 但这些表征在 OOD 下普遍不稳

所以更像：

> **physics-flavored representation**，而不是 **robust physics world model**。

---

# 下一步该怎么做？

1. 在 bottleneck 上加入 **显式动力学辅助损失**
2. 做真正的 **counterfactual intervention**
3. 提升 two-body 评测，从中心量升级到对象级守恒量
4. 引入更强的局部归纳偏置，而不是单纯更大模型

---

# Backup 1：扩展实验矩阵

本次已跑：

- 5 个运动族：circular / projectile / bounce / pendulum / two-body
- 3 个模型族：GRU / local-attention / bottleneck
- 共 **16** 组实验

命名格式：

- `{motion}_{model}_{label}`

例如：

- `circular_gru_long`
- `pendulum_bottleneck_short`

---

# Backup 2：最关键的定量结论

## 最值得记住的 4 句话

1. **GRU-long 在 ID 上最好**
2. **OOD 并没有随之变好**
3. **local-attention 并没有自动更物理**
4. **state bottleneck 也没有单独解决问题**

---

# Backup 3：材料位置

- 报告：`docs/report.md`
- 中文 slides：`docs/slides_cn.md`
- 总结果：`artifacts/summary.json`
- 总表：`docs/assets/tables/table_quantitative_results.csv`
- 图：`docs/assets/figures/`
- GIF：`docs/assets/videos/`

---

# Thank you

讨论问题：

1. 物理 world model 的最低必要条件是什么？
2. OOD probe 是否应该成为“物理涌现”的必要标准？
3. 下一步最值得加的是更强模型，还是更强归纳偏置？
