# Feasible Energy Transport Generation (FETG)

## A research note and method draft

> Status: exploratory method draft grounded in local toy demos, literature search, and honest novelty constraints.
> 
> Claim level: **framework proposal / architecture proposal**, not a validated new algorithmic family yet.

---

## 1. Proposed Method Name

**Feasible Energy Transport Generation (FETG)**

Chinese shorthand:

**可行域-能量-输运生成**

The core idea is to generate in an **interpretable physical state space** rather than in pixel space or an opaque latent particle space, while combining:

1. **feasible-state proposals**,
2. **energy-based refinement**,
3. **transport-aware state matching**, and
4. **rendering as observation, not as the generated object itself**.

---

## 2. Motivation

Most mainstream generative models for images and videos operate in one of two modes:

- **diffusion / denoising** in a fixed observation space,
- **flow matching / vector field learning** in a continuous latent or signal space.

These are powerful, but they are often misaligned with problems where the true object of interest is not the rendered observation, but an underlying **physical state**.

Examples:

- a cup with valid geometry, capacity, and stability,
- a multi-body configuration with non-penetration constraints,
- a particle/system state that should satisfy conservation or feasibility rules,
- a design object where the image is only a rendering of geometry and material parameters.

In these settings, two problems appear repeatedly:

1. **Pixel-space similarity is physically weak.**
   Two states may render to shifted observations that are perceptually close but have large pixelwise error.

2. **Feasibility is often treated as an afterthought.**
   Constraints are commonly injected by guidance, penalties, or projection after a generic generator has already proposed states.

FETG is a proposal to make the **physical state** the primary object, and the image only a secondary observation.

---

## 3. Problem Definition

We assume an underlying state:

\[
s \in \mathcal{S}
\]

where \(\mathcal{S}\) is a structured physical/design state space, for example:

- geometry parameters,
- material parameters,
- topology variables,
- field variables,
- object configurations.

We further define:

- a **feasible set** \(\mathcal{F} \subseteq \mathcal{S}\),
- an **energy** \(E(s)\) encoding physical plausibility or design preference,
- an **observation model / renderer** \(R(s)\) producing image-like or sensor-like outputs.

Observed data are:

\[
x = R(s)
\]

possibly with noise, partial observability, or ambiguity.

The goal is not merely to model \(p(x)\), but to model or sample states \(s\) such that:

1. \(s \in \mathcal{F}\),
2. \(E(s)\) is low,
3. rendered observations \(R(s)\) match data,
4. state distributions match the target distribution under a geometry-aware or transport-aware metric.

---

## 4. Core Proposal

FETG proposes a generative process over state space with the following structure:

```text
noise / seed
   ↓
feasible proposal mechanism
   ↓
energy-based refinement in state space
   ↓
transport-aware matching objective
   ↓
renderer / observation head
```

### 4.1 Feasible proposal mechanism

Rather than sampling arbitrary states and hoping constraints are repaired later, start from a proposal distribution supported on the feasible set or near it:

\[
s_0 \sim q_0(s), \quad \text{with} \quad q_0 \text{ concentrated on } \mathcal{F}
\]

Possible implementations:

- direct constraint-native samplers,
- projection-based samplers,
- factorized constructive samplers,
- state parameterizations that encode feasibility by design.

### 4.2 Energy-based refinement

Refine the proposal in state space using an explicit energy:

\[
E(s) = E_{\text{feas}}(s) + E_{\text{physics}}(s) + E_{\text{design}}(s) + E_{\text{regularity}}(s)
\]

Sampling can be done with Langevin-style dynamics:

\[
s_{k+1} = \Pi_{\mathcal{F}}\left(s_k - \eta \nabla E(s_k) + \sigma \epsilon_k\right),
\quad \epsilon_k \sim \mathcal{N}(0, I)
\]

where \(\Pi_{\mathcal{F}}\) is optional hard projection or an implicit feasible parameterization.

### 4.3 Transport-aware state matching

Instead of relying primarily on pixelwise losses, compare generated and target states using transport-aware distances in state space:

\[
\mathcal{L}_{\text{transport}} = W_c\big(\mu_{\theta}, \mu_{\text{data}}\big)
\]

where:

- \(W_c\) is a Wasserstein / OT-style metric,
- \(c(s, s')\) is a geometry-aware state cost,
- \(\mu_{\theta}\) and \(\mu_{\text{data}}\) are generated and target state distributions.

This is meant to reward correspondence in **physical state space**, not just observation space.

### 4.4 Rendering as observation

The renderer is not the generator itself. It is an observation map:

\[
x = R(s)
\]

The rendering loss is auxiliary / observation-aligned:

\[
\mathcal{L}_{\text{render}} = d\big(R(s), x^*\big)
\]

for some image or observation-space metric \(d\).

The key philosophical point is:

> **the model generates states, not images**.

---

## 5. Full Objective (Method Draft)

One plausible training objective is:

\[
\mathcal{L}(\theta)
=
\lambda_1 \mathcal{L}_{\text{transport}}
+ \lambda_2 \mathcal{L}_{\text{render}}
+ \lambda_3 \mathcal{L}_{\text{energy}}
+ \lambda_4 \mathcal{L}_{\text{feasibility}}
\]

with:

- **transport loss**: aligns generated and target state distributions,
- **render loss**: keeps rendered observations faithful,
- **energy loss**: encourages low-energy states,
- **feasibility loss**: optional soft penalty if feasibility is not entirely hard-coded.

One concrete choice is:

\[
\mathcal{L}_{\text{energy}} = \mathbb{E}_{s \sim q_\theta}[E(s)]
\]

and:

\[
\mathcal{L}_{\text{feasibility}} = \mathbb{E}_{s \sim q_\theta}[\phi(s)]
\]

where \(\phi(s)=0\) on the feasible set and positive outside.

At this stage, this is a **framework-level mathematical sketch**, not yet a finalized algorithm.

---

## 6. Relation to Existing Paradigms

### 6.1 Diffusion

**Diffusion** typically learns denoising in an observation-like space or latent signal space.

FETG differs in emphasis:

- generation target is the **state** rather than the image,
- feasibility is treated as native structure rather than guidance-only correction,
- energy is explicit and interpretable,
- transport matching is proposed in state space rather than pixel space.

### 6.2 Flow Matching

**Flow matching** learns a vector field connecting source and target distributions.

FETG differs by not requiring the main story to be “learn a velocity field.” Instead, it focuses on:

- feasible proposals,
- energy-driven local refinement,
- state-space distribution matching.

Flow matching could still appear as a subcomponent, but it is not the defining primitive.

### 6.3 Projected Diffusion / Constrained Diffusion

Methods like projected diffusion, constrained diffusers, and related constrained sampling methods already show that hard constraints can be introduced into diffusion-like processes.

FETG should **not** claim to invent constraint-aware generation.

Its possible distinction is narrower:

- no commitment to diffusion as the base process,
- explicit physical/design state parameterization,
- explicit energy shaping,
- proposed transport-aware matching directly over states,
- rendering treated as observation instead of the generated object.

### 6.4 Inverse Design

Inverse design methods often optimize latent codes or parameters to satisfy downstream objectives.

FETG is close to inverse design, and must acknowledge that.

The intended difference is that FETG is framed as a **generative state-space modeling pipeline**, not only a deterministic optimizer:

- it aims to model families/distributions of valid states,
- it includes stochastic feasible proposals,
- it includes explicit distribution matching in state space.

Still, reviewer pushback of the form “this is just inverse design in another wrapper” should be expected.

---

## 7. Local Feasibility Evidence

This note is not purely speculative; it is grounded in two local toy demonstrations.

### 7.1 Demo 4: Constraint-native 2D generation

Path:

`physics_state_gen_lab/demo4_constrained_2d`

Observed local metrics:

- unconstrained validity: **85.0%**
- projected validity: **98.0%**
- constraint-native validity: **100.0%**

Takeaway:

> hard feasibility by construction can outperform unconstrained or post-projected sampling in a toy geometric setting.

### 7.2 Demo 5: Energy-based cup-like state generation

Path:

`physics_state_gen_lab/demo5_energy_shapes`

Observed local metrics:

- random valid ratio: **0.75**
- refined valid ratio: **1.0**
- mean energy: **8.39 → 0.52**

Takeaway:

> explicit energy shaping can refine interpretable state parameters toward more plausible and more valid states.

### 7.3 Demo 6: Transport-aware state matching

Path:

`physics_state_gen_lab/demo6_transport_state_matching`

Observed local metrics (from automated tests):

- pixel metric symmetry: **verified**
- aligned-transport metric symmetry: **verified**
- translated-same-shape distance < different-shape distance: **verified**

Key property demonstrated:

- When the same shape (e.g., circle) is translated to different positions, the centroid-aligned transport distance is **lower** than the distance between different shapes (e.g., circle vs square).
- This is not true of pixel-space L2: the translated circle can have **higher** pixel distance to another circle than to a different shape, depending on positions.

Takeaway:

> in a toy 2D point-cloud setting, transport-aware metrics better capture state identity under translation than pixel-space metrics. This provides proof-of-concept evidence for the third FETG pillar.

**Important caveat:** this is a low-dimensional toy validation. The property does not automatically scale to high-dimensional image spaces or complex state manifolds. High-dimensional optimal transport remains expensive and brittle (see Section 9.2).

### 7.4 Demo 7: Rendering as observation

Path:

`physics_state_gen_lab/demo7_renderer_observation`

Observed local metrics (from automated tests):

- renderer output shape: **verified** (64x64 images from point sets)
- rendered image changes when state changes: **verified**
- naive recovery returns valid state within expected bounds: **verified**
- recovery on masked observation beats naive baseline: **verified**

Key properties demonstrated:

- A simple 2D point-set renderer maps state → observation (grayscale image)
- A naive centroid-based recovery mechanism can reconstruct state from observation
- On partially masked observations, the recovery mechanism achieves lower image-space distance than a random baseline

**Important caveat:** this is a low-dimensional toy validation. It does **NOT** demonstrate:
- differentiable rendering
- learned inverse graphics
- scalability to high dimensions
- real inverse problems

Takeaway:

> in a toy 2D point-cloud setting, a simple render→observe→recover loop can achieve non-trivial recovery quality. This provides proof-of-concept evidence for the fourth FETG pillar ("rendering as observation"), with explicit acknowledgment that this does not prove the full inverse graphics pipeline.

### 7.5 Demo 8: Integrated FETG Pipeline

Path:

`physics_state_gen_lab/demo8_integrated_fetg_pipeline`

Observed local metrics (from automated tests):

- module imports: **verified**
- feasible proposal returns valid states: **verified** (shape, bounds, feasibility check)
- renderer output shape: **verified** (64x64 images from point sets)
- integrated energy is finite for valid states: **verified**
- refinement improves energy or observation loss relative to baseline: **verified**

Key properties demonstrated:

- Combines all four FETG pillars in a single unified pipeline: feasible proposals → energy refinement → transport metrics → renderer/observation
- On a fixed seeded 2D point-cloud case, energy-based refinement reduces energy or observation loss vs. the feasible starting proposal
- The baseline (random unconstrained proposal) comparison shows the effect of starting from a better proposal distribution

**Important caveats (non-exhaustive):**

- This is a toy 2D point-cloud state space only. It does **NOT** demonstrate:
  - scalability to high-dimensional physical state spaces
  - learned transport matching (only centroid-aligned, not learned)
  - differentiable rendering (only raster, not differentiable)
  - real-world physical constraint satisfaction
  - anything beyond the four pillars composed naively in sequence

Takeaway:

> demo8 is the **first integrated FETG pipeline** after demos 4-7. It demonstrates that the four pillars can be composed in a single run, and that refinement improves at least one of (energy, observation loss) vs. the feasible starting proposal. This is a **necessary but not sufficient** condition for the broader FETG claim — it does not validate scalability, learned components, or real-world applicability.

---

With demo8, we have the **first integrated pipeline** combining all four intended FETG pillars in sequence. Demos 4-7 remain as independent pillar validations. Demo8 does not replace or subsume them; it shows the pillars can work together in a toy setting.

1. feasible proposal structure (demo4),
2. energy-based refinement (demo5),
3. transport-aware state matching (demo6),
4. rendering as observation (demo7).

Each remains a low-dimensional toy validation. The full FETG framework is not yet validated.

---

## 8. Minimal Experiment Plan

The most realistic minimal plan is to validate the framework in stages.

### Stage A — Constraint + energy integration

**Goal:** unify demo4 and demo5.

Experiment:

- use a feasible proposal sampler rather than random initialization,
- perform energy refinement on top of feasible proposals,
- compare against:
  - random init + refinement,
  - projected init + refinement,
  - unconstrained baselines.

Success criteria:

- better validity,
- lower energy,
- better diversity than purely projected methods.

### Stage B — Transport-aware state matching

**Goal:** validate the third pillar.

Experiment:

- create a toy task where pixelwise similarity is misleading,
- compare:
  - pixel L2,
  - feature loss,
  - transport-aware state matching.

Candidate toy benchmarks:

- moving blob / shape translation,
- point-cloud object relocation,
- cup profile families with shifted but semantically equivalent geometry.

Success criteria:

- transport-aware objective better correlates with state identity than pixel loss,
- better recovery under shifts, correspondences, or mild topology-preserving deformation.

### Stage C — Rendering as observation

**Goal:** close the state → render loop.

Experiment:

- define a simple renderer for 2D states,
- train/evaluate under partial observation,
- reconstruct states via render supervision.

Success criteria:

- latent state remains interpretable,
- rendered output matches observation,
- state-space metrics remain meaningful.

### Stage D — Reviewer-facing baseline comparison

Compare against at least:

- a simple diffusion baseline,
- a flow-matching-style baseline,
- a projected/constrained sampling baseline,
- a pure energy-based baseline.

Without this stage, novelty claims will remain weak.

---

## 9. Main Risks

### 9.1 This may be “just a composition”

The strongest reviewer objection is likely:

> “This is not a new algorithm; it is a composition of constrained generation, energy refinement, OT-style matching, and rendering.”

That objection is partly correct. The current safest claim is indeed at the **framework / architecture level**.

### 9.2 OT scaling may fail in practice

Transport-aware metrics are conceptually attractive, but high-dimensional OT is expensive and brittle.

This is the most likely technical bottleneck in turning the framework into a practical method.

### 9.3 Feasibility may not scale

Constraint-native generation looks strong in low-dimensional toy spaces, but nonconvex or high-dimensional feasible sets can break the story quickly.

### 9.4 Energy landscapes may collapse diversity

Energy refinement may simply funnel samples into a few low-energy basins rather than representing a rich target distribution.

### 9.5 Renderer gradients may be weak or misleading

If rendering is used as observation supervision, the renderer may become the bottleneck instead of the state model.

---

## 10. Honest Novelty Claim Wording

### Safe wording

Use phrasing like:

- “We propose a **state-space native generative framework**...”
- “We explore a **pipeline architecture** combining feasible proposals, energy refinement, transport-aware state matching, and rendering-based observation...”
- “We provide **proof-of-concept toy evidence** for two core ingredients...”
- “Our contribution is primarily a **unified formulation and experimental hypothesis**, rather than a fully validated algorithmic family.”

### Unsafe wording

Avoid phrases like:

- “first method to...”
- “new generative paradigm”
- “strictly more general than diffusion / flow matching”
- “state-of-the-art”
- “solves constrained physical generation”

These claims are not currently defensible.

---

## 11. Recommended Positioning

If written today, the strongest truthful positioning is:

> FETG is a **research hypothesis and framework draft** for physical state generation, motivated by the mismatch between observation-space generation and state-space design problems.
> 
> It is not yet a proven new generative family. Its current contribution is the proposal to unify feasibility, energy, transport, and observation into a single state-centered formulation, supported by preliminary toy evidence.

This is still useful.

Many good papers begin as exactly this kind of well-posed framework note, then become strong once a decisive third experiment validates the missing pillar.

---

## 12. Immediate Next Step

The next decisive step is:

> **implement the transport-aware state-matching experiment**

Without that, FETG is missing its third pillar and risks collapsing into “constraint sampler + EBM refinement.”

With that, it becomes much easier to argue that the framework is doing something structurally different from both diffusion-style denoising and standard inverse design.
