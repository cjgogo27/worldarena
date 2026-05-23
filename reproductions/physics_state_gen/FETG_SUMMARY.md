# FETG: 从“生成图片”到“生成状态”

Created in April 2026

---

## Motivation

这组实验的出发点很简单：如果我们真的想做一种不同于 diffusion / flow matching 的生成方法，那么我们也许不应该再把“生成”理解为**直接在像素空间里把噪声变成图像**，而应该把它理解为：

> **先生成一个可解释的 state，再把图像当成这个 state 的 observation。**

这个想法来自几个持续反复出现的直觉：

1. **很多视觉变化本质上是 state 的变化，而不是像素值在原地重写。**
   一个物体平移了，我们直觉上会说“它移动了”，而不是“原位置消失、目标位置长出来”。

2. **pixel-wise loss 往往不是合适的比较方式。**
   如果同一个形状平移了几像素，pixel L2 可能很大，但从 state 角度它其实几乎没变。

3. **物理或几何约束常常应该是生成的一部分，而不是事后修补。**
   如果一个状态必须满足不重叠、稳定、边界合法，那么最自然的做法是直接在可行域里工作。

因此，我们把整条探索线暂时组织成一个框架草案：

## FETG = Feasible + Energy + Transport + Rendering

也就是四根柱子：

1. **Feasible proposals**：先提出合法状态；
2. **Energy refinement**：再用能量把状态往更合理的地方推；
3. **Transport-aware matching**：比较状态时不要只看像素；
4. **Rendering as observation**：图像是观测，不是生成本体。

---

## Observation 1: Feasibility by construction is stronger than repair after the fact

第一个问题是：如果状态有明确约束，那么应该怎么生成？

在 `demo4_constrained_2d` 里，我们选了一个极小的 toy 问题：两个二维圆盘，要求它们**不重叠**且**落在边界内**。然后比较三种方式：

- **unconstrained**：直接高斯采样；
- **projected**：先采样，再投影回合法区域；
- **constraint-native**：直接在合法状态集合中采样。

结果非常直接：

| method | validity rate |
|---|---:|
| unconstrained | 0.85 |
| projected | 0.98 |
| constraint-native | 1.00 |

我们还看到，constraint-native 样本在 clearance 上也更健康：

- unconstrained 最小 clearance：`-0.9087`
- projected 最小 clearance：约 `0`
- constraint-native 最小 clearance：`0.0020`

这说明一个很朴素的事实：

> 如果约束本身是问题定义的一部分，那么“先乱采样再修”并不是最自然的生成逻辑。

这并不意味着 projected 方法没有用，而是说：**feasibility 可以成为生成本体的一部分，而不只是后处理。**

---

## Observation 2: Energy refinement can improve plausibility even without diffusion or flow

第二个问题是：如果我们已经有了一个 state，接下来怎样把它往“更合理”的地方推？

在 `demo5_energy_shapes` 里，我们用了一个简化的 cup-like state：

- width
- height
- base thickness
- wall thickness
- handle size

然后定义一个 cup energy，用来惩罚不合理几何或不稳定参数组合，再用 Langevin-style refinement 推状态。

结果也很清楚：

| metric | before | after |
|---|---:|---:|
| valid ratio | 0.75 | 1.00 |
| mean energy | 8.386 | 0.519 |

换句话说：

> 即使不依赖 diffusion 的去噪路径，也可以通过显式 energy 把状态推向更 plausible 的区域。

这当然还只是一个 toy。它没有证明 energy-based 方法能自动扩展到高维复杂视觉空间，但它至少说明：

- state 可以被显式定义；
- plausibility 可以被显式写成 energy；
- refinement 可以直接在 state space 内发生。

---

## Observation 3: Pixel similarity is often the wrong comparison; state-aware matching can be more faithful

第三个问题是：如果我们生成的是 state，那状态之间该怎么比较？

`demo6_transport_state_matching` 用了几个 canonical shapes：

- triangle
- square
- circle
- L-shape

然后对每个 shape 做平移，比较两种 metric：

1. **pixel L2**：先 rasterize，再比较图像；
2. **centroid-aligned transport-style distance**：先去掉整体平移，再比较 shape-relative point sets。

这个 demo 的一个关键诚实点是：

> 我们并没有声称“普通 OT 天然平移不变”。

这里验证的是一个更有限的命题：**一个 centroid-aligned 的 transport-aware distance，在这个 toy setting 下更接近 shape identity。**

例如：

- triangle 的 pixel range 约为 `0.449`
- triangle 的 aligned transport range 约为 `2.95e-15`

对所有 shape 都类似：

- pixel 距离对平移很敏感；
- aligned transport 距离几乎不变。

这说明：

> 如果生成对象本来就是 state，那么比较它们时最好也在 state space 中比较，而不是强迫所有东西都投影成像素再比较。

---

## Observation 4: Rendering should be treated as observation, not as the generated object itself

第四个问题是：如果 state 才是本体，那图像应该扮演什么角色？

`demo7_renderer_observation` 给出的回答是：

> 图像是 observation。

在这个 demo 中，我们构造 point-set state，做：

```text
state -> render -> image -> mask/noise -> recover state
```

这不是一个强大的 inverse graphics 系统，它只是一个 toy recovery loop。但它足以说明概念：

- 我们可以从 state 生成 observation；
- observation 可以是部分的、带 mask 的；
- 恢复的目标不是“复原每个像素”，而是“找回潜在 state”。

一个最直接的 masked recovery 结果是：

- naive masked distance: `4.929`
- iterative masked distance: `4.714`
- improvement: `0.215`

这个 improvement 不大，但已经足够说明：

> render → observe → recover 这条链条是可以闭起来的。

也就是说，图像可以被视为 state 的观测，而不一定是生成本体本身。

---

## Observation 5: The four pillars can be composed, but the objectives already compete

做完前四个 demo，最自然的问题就是：

> 它们能不能在一个 pipeline 里一起工作？

`demo8_integrated_fetg_pipeline` 是第一次把四根柱子串起来：

```text
true state
  -> render
  -> corrupted observation
  -> visible hints
  -> feasible proposal
  -> energy refinement
  -> transport-aware comparison
  -> best recovered state
```

结果既鼓舞人，也很诚实。

### Feasible proposal -> refined

- energy: `0.960 -> 1.322`
- observation loss: `20.469 -> 19.545`

### Random baseline -> refined

- energy: `3.380 -> 1.774`
- observation loss: `21.119 -> 21.656`

这说明两件事：

1. **四根柱子是可以串起来跑的。**
2. **多目标之间已经开始打架了。**

在 integrated pipeline 里，refinement 并没有让所有指标同时变好。它改善了 observation fit，却让 integrated branch 的 energy 变差。这不是坏消息，反而是一个真实信号：

> 一旦把 feasibility、energy、transport、rendering 放进同一个 objective，权重平衡就会立刻成为核心问题。

所以 demo8 的意义不是“FETG 已经成立”，而是：

> 它第一次证明了这四根柱子不是只能各自成立，而是可以在一个 toy pipeline 中同时出现。

---

## What these demos actually validate

把所有结果合在一起，我们现在能诚实地说：

### 已验证的

1. **feasible proposals** 是有价值的；
2. **energy refinement** 能把 state 推向更合理区域；
3. **transport-aware matching** 在某些 toy identity 问题上比 pixel L2 更合理；
4. **rendering as observation** 可以作为 state-centered pipeline 的一部分；
5. 这四件事可以被组合成一个最小 integrated pipeline。

### 尚未验证的

这些 toy demos **没有**证明：

- FETG 已经能替代 diffusion；
- 它能生成复杂自然图像或视频；
- 它能扩展到高维 state spaces；
- transport term 在真实高维场景仍然可用；
- rendering loop 可以自然扩展到真正的 inverse graphics；
- 我们已经拥有一个稳定、可训练、可扩展的生成 family。

所以当前最安全的结论应该是：

> 这是一条**值得继续推进的 state-centered generative route**，但它现在仍然只是一个 framework hypothesis backed by toy evidence。

---

## A provisional picture of FETG

如果把这些 demo 暂时压缩成一个统一图景，它看起来像这样：

```text
noise / seed
  -> feasible proposal in state space
  -> energy-based refinement
  -> transport-aware state comparison
  -> renderer / observation
```

在这个视角里，图像不再是生成对象本身，而是：

\[
x = R(s)
\]

其中：

- \(s\) 是可解释 state；
- \(R\) 是 renderer / observation map；
- 生成系统真正操作的是 \(s\)，而不是 \(x\)。

这正是我们想和 diffusion / flow matching 拉开距离的地方：

- diffusion / flow matching 更像是在 observation-like space 上学路径；
- FETG 更像是在 **state space 上组织 proposal, refinement, comparison, observation**。

---

## What I currently believe

经过这组 toy experiments，我现在最相信的不是“我们已经找到了一种可以取代 diffusion 的方法”，而是更克制的一句话：

> 对于一类 genuinely state-centric 的问题，直接生成 observation 可能不是最合适的建模方式；把 state 当成本体，把 image/video 当成 observation，可能会带来更自然的结构与约束接口。

这条路真正吸引人的地方，不在于它今天已经跑赢了 diffusion，而在于它有机会把下面这些东西放进同一个生成框架：

- 几何约束
- 物理约束
- state-level identity
- observation-level supervision
- inverse design
- structured editing

而这恰恰是很多纯 observation-space 生成方法不太自然的地方。

---

## Immediate next questions

如果继续往前走，我觉得最重要的不再是再加一个独立 toy demo，而是回答 integrated setting 里的几个真实问题：

1. **Objective balancing**  
   为什么 observation loss 能改善，但 energy 变差？怎样平衡 feasibility / energy / transport / render？

2. **Learned proposals**  
   现在 proposal 还比较手工，什么时候应该引入 learned proposal model？

3. **State scaling**  
   state 从 circle / point cloud / cup parameters 扩展到更复杂对象后，还能不能保持可解释性和 tractability？

4. **Dynamics / video**  
   如果把 state 再加上 dynamics，是否真的能比 video diffusion 更自然地表达 identity persistence、birth/death、occlusion？

这些问题都还没解决。但至少现在，我们已经有了一条相对清晰的实验路线，而不只是一个空洞的想法。

---

## Code

Relevant local paths:

- `physics_state_gen_lab/demo4_constrained_2d/`
- `physics_state_gen_lab/demo5_energy_shapes/`
- `physics_state_gen_lab/demo6_transport_state_matching/`
- `physics_state_gen_lab/demo7_renderer_observation/`
- `physics_state_gen_lab/demo8_integrated_fetg_pipeline/`
- `physics_state_gen_lab/RESEARCH_NOTE_FETG.md`

---

## Final takeaway

如果只用一句话总结这条探索线，我会写成：

> **We may not want to generate images directly. We may want to generate states, and let images be observations.**

现在这还只是一个 toy-supported research direction，而不是已被证明的新生成范式。但它已经足够具体，值得继续往下做了。
