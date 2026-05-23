# Notes

## Research Intent

Explore whether physics-inspired state generation can complement or outperform diffusion/flow approaches on constrained/energy minimization tasks.

## Hypotheses

- Hard constraints may produce more physically plausible states than learned denoising
- Energy minimization can capture equilibrium states that diffusion models struggle with
- These approaches may generalize better to tasks with explicit physical rules (boundaries, conservation)

## What to Watch For

- `demo4_constrained_2d`: Does constraint enforcement produce more stable dynamics than soft constraints in diffusion? Watch for boundary violations.
- `demo5_energy_shapes`: Does energy minimization converge to plausible shapes? Watch for local minima traps, convergence speed.

## Next Experiments

1. **Transport-aware metrics**: Measure Wasserstein distance between generated and target distributions, not just pixelwise loss
2. **Birth/death dynamics**: Add particle creation/destruction to model state transitions (e.g., phase changes)
3. **Renderer integration**: Connect energy-based states to differentiable renderer for inverse graphics
4. **Cup design**: Test energy-based approach on 3D cup/container shape optimization with physics constraints

## Folder Structure

```
physics_state_gen_lab/
├── README.md
├── NOTES.md
├── demo4_constrained_2d/   # Constrained generation
├── demo5_energy_shapes/     # Energy-based generation
├── demo6_transport_state_matching/ # Transport-aware metrics (proof-of-concept)
├── demo7_renderer_observation/      # Renderer-observation loop (fourth FETG pillar)
└── demo8_integrated_fetg_pipeline/ # First integrated pipeline combining all four pillars
```

## Demo 6: Transport-Aware Metrics

Added as proof-of-concept validation of the third FETG pillar (transport-aware state matching). Key tests verify:

- Pixel metric and aligned-transport metric are symmetric
- Translated same-shape pairs have lower aligned-transport distance than different-shape pairs

This remains a **toy validation**: high-dimensional OT scaling is not demonstrated.

## Demo 7: Renderer-Observation Loop

Added as proof-of-concept validation of the fourth FETG pillar ("rendering as observation"). Key tests verify:

- Renderer output shape is correct (64x64 images)
- Rendered image changes when state changes
- Naive recovery returns valid state within expected bounds
- Recovery on masked observation beats naive baseline

**Important caveat:** This does NOT prove differentiable rendering, real inverse graphics, or scalability. It is a low-dimensional toy validation only.

## Demo 8: Integrated FETG Pipeline

Added as the **first integrated FETG pipeline** combining all four pillars in a single unified run: feasible proposals (demo4), energy refinement (demo5), transport metrics (demo6), and renderer/observation (demo7).

Automated tests verify:
- All demo8 modules can be imported
- Feasible proposal sampler returns valid states (shape, bounds, feasibility check)
- Renderer output shape is correct (64x64 images)
- Integrated energy is finite for valid states
- Refinement improves energy or observation loss relative to a seeded baseline

**Important caveat:** Toy 2D point-cloud integration only. Does NOT prove scalability, learned transport, differentiable rendering, or high-dimensional success. This is a necessary but not sufficient condition for the broader FETG claim.
