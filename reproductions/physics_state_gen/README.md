# Physics State Generation Lab

Sandbox for exploring alternative state generation paradigms outside diffusion/flow matching.

## Why This Matters

Diffusion and flow matching learn to denoise or interpolate between noise and data. These demos explore physics-inspired alternatives where state evolves through constraints or energy minimization rather than learned transformations.

## Demos

- `demo4_constrained_2d/`：Constrained 2D generation — state evolves under hard constraints (e.g., boundary conditions, conservation laws) rather than probabilistic denoising
- `demo5_energy_shapes/`：Energy-based shape generation — state emerges from energy minimization (e.g., spring networks, potential fields) rather than flow
- `demo6_transport_state_matching/`：Transport-aware state matching — centroid-aligned transport metric is translation-invariant (unlike naive pixel L2)
- `demo7_renderer_observation/`：Renderer-observation loop — state renders to image; naive recovery from masked observation beats baseline (toy evidence for "rendering as observation")
- `demo8_integrated_fetg_pipeline/`：**first integrated FETG pipeline** combining all four pillars above (feasible proposals, energy refinement, transport matching, rendering as observation). **Toy 2D integration only** — does NOT prove scalable or learned FETG.

## Quick Run

```bash
python physics_state_gen_lab/demo4_constrained_2d/run.py
python physics_state_gen_lab/demo5_energy_shapes/run.py
python physics_state_gen_lab/demo6_transport_state_matching/run.py
python physics_state_gen_lab/demo7_renderer_observation/run.py
python physics_state_gen_lab/demo8_integrated_fetg_pipeline/run.py
```

If your default `python` environment lacks numpy/matplotlib, use the project's working env instead.

## Smoke Tests

```bash
python physics_state_gen_lab/tests/run_smoke_tests.py
```

There is also a pytest-style test file under `physics_state_gen_lab/tests/`, but the smoke runner avoids requiring pytest in thin environments.

Each demo generates:
- `outputs/*.png`：Visualization
- `outputs/metrics.json`：Key metrics

## Dependencies

- Python 3.9+
- numpy
- matplotlib
