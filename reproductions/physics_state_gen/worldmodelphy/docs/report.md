# World Models for Physics: A Research Report

> A technical investigation into whether small video generation models learn physical reasoning from scratch.

This report consolidates experimental findings from this repository. For the literature synthesis that motivated the experiments, see [`research_summary.md`](./research_summary.md).

---

## Motivation

Do video generative models trained from scratch on synthetic motion sequences develop genuine physical understanding, or do they merely learn to interpolate visual patterns? This matters because recent large-scale work repeatedly shows that visual realism and physical reasoning are not the same thing.

This repository tests a narrow version of that question under tight control:

- simple one-ball videos
- fully known generating physics
- tiny models trained from scratch
- explicit in-distribution and out-of-distribution evaluation

The central question is whether changing **context length** shifts the learned representation toward local dynamics or toward global trajectory templates.

---

## Literature Synthesis

This work builds directly on three observations from recent literature:

1. **From Kepler to Newton** (arXiv:2602.06923): prediction accuracy alone does not imply physics understanding; temporal locality can push models from global curve-fitting toward local dynamics.
2. **Physics-IQ** (arXiv:2501.09038): strong visual realism can coexist with weak physical understanding.
3. **Morpheus** (arXiv:2504.02918): current video generative models still struggle with conservation-law-grounded plausibility.

The working hypothesis here was:

- short-context training may encourage local dynamics
- long-context training may improve in-distribution quality but overfit trajectory templates

---

## Experimental Setup

### Research Questions

1. Does short-context training promote local dynamics encoding?
2. Does long-context training lead to trajectory-template learning?
3. Do latent linear probes reveal genuine physics, or only training-manifold memorization?

### Evaluation Metrics

- **Frame MSE**: pixel-space rollout error
- **Trajectory MSE**: centroid-space object motion error
- **Velocity / acceleration MSE**: first- and second-difference trajectory errors
- **Linear probe R²**: whether latent states linearly expose `vx`, `vy`, `ax`, `ay`

### Architecture Choices

The implemented model is not a transformer. It is a tiny recurrent video model:

- **Frame encoder**: three strided convolutions
- **Temporal core**: GRU with hidden size 128
- **Frame decoder**: transposed convolutions

Two variants were trained:

| Variant | Context length | Hidden size | Latent size | Epochs |
|--------|----------------|-------------|-------------|--------|
| Short-context | 4 | 128 | 64 | 8 |
| Long-context | 16 | 128 | 64 | 8 |

Teacher forcing was annealed from 1.0 to 0.56.

---

## Datasets

The data is fully synthetic and generated on the fly.

| Property | Description |
|----------|-------------|
| Domain | Circular motion, projectile motion |
| Resolution | 64×64 grayscale |
| Sequence length | 32 frames |
| Train / val / test / OOD | 96 / 24 / 24 / 24 sequences per motion |
| Stored metadata | position, velocity, acceleration, generating parameters |

### OOD shifts

- **Circular**: larger radius, higher angular velocity, shifted center
- **Projectile**: different gravity, faster launch speed, different initial position

This setup is intentionally simple: if physics does not emerge here, it is unlikely to emerge reliably in a harder pixel-space setting without stronger inductive bias.

---

## Models

The same CNN+GRU+CNN predictor is trained under two context regimes.

| Model | Context Length | Training Objective |
|-------|----------------|-------------------|
| `circular_short` / `projectile_short` | 4 | autoregressive frame reconstruction |
| `circular_long` / `projectile_long` | 16 | autoregressive frame reconstruction |

Because the architecture is fixed, this isolates the effect of **how much trajectory history** the model sees during training.

---

## Experiments

### Experiment 1: In-Distribution Learning

Train on synthetic videos and evaluate on held-out test sequences from the same parameter range.

### Experiment 2: OOD Generalization

Roll out from contexts drawn from unseen physical parameters.

### Experiment 3: Latent Linear Probes

Fit linear least-squares probes from GRU hidden states to:

- `vx`
- `vy`
- `ax`
- `ay`

Train probes on the training split, then evaluate on test and OOD splits.

---

## Results

### Quantitative Results

| Motion | Metric | Short-context | Long-context |
|--------|--------|---------------|--------------|
| Circular | ID trajectory MSE | 23.20 | 3.15 |
| Circular | OOD trajectory MSE | 787.68 | 583.65 |
| Circular | ID probe mean R² | 1.00 | 1.00 |
| Circular | OOD probe mean R² | -0.04 | -0.01 |
| Projectile | ID trajectory MSE | 38.49 | 5.76 |
| Projectile | OOD trajectory MSE | 307.59 | NaN (catastrophic collapse) |
| Projectile | ID probe mean R² | 1.00 | 1.00 |
| Projectile | OOD probe mean R² | -24.44 | -21.23 |

Raw table:

- `./assets/tables/table_quantitative_results.csv`

### Main Pattern

There is a clear split between **in-distribution competence** and **out-of-distribution physics**:

- Long-context models are better on **ID trajectory quality**.
- None of the models are good on **OOD physics**.
- ID linear probes are almost perfect for all runs.
- OOD linear probes collapse sharply, often becoming strongly negative.

That combination strongly suggests **training-manifold encoding**, not robust physical law discovery.

### Visual Results

- ID comparison strip: `./assets/figures/fig_id_generation_comparison.png`
- ID sample GIF: `./assets/videos/gif_id_samples.gif`
- OOD failure plot: `./assets/figures/fig_ood_failures.png`
- OOD failure GIF: `./assets/videos/gif_ood_breakdown.gif`
- Probe summary figure: `./assets/figures/fig_probe_accuracy.png`

Per-experiment loss curves:

- `./assets/figures/fig_loss_curve_circular_short.png`
- `./assets/figures/fig_loss_curve_circular_long.png`
- `./assets/figures/fig_loss_curve_projectile_short.png`
- `./assets/figures/fig_loss_curve_projectile_long.png`

---

## Failure Modes

### 1. Long-context template learning

Long context improves in-distribution rollout quality, but does not solve OOD physics. This is exactly the pattern expected from stronger trajectory-template fitting.

### 2. Probe optimism on ID data

All four runs achieve near-perfect probe R² on test data from the same distribution. If we stopped there, we might incorrectly conclude that physics had emerged.

OOD probes show the opposite: the hidden states do not transfer cleanly once generating parameters move off the training manifold.

### 3. Catastrophic rollout collapse

`projectile_long` failed so hard on OOD rollout that centroid extraction returned no finite trajectory, producing `NaN` rollout metrics. This is a useful result, not just a nuisance: it shows how brittle visually-trained predictors can be under parameter shift.

### 4. Appearance success hiding dynamics failure

Frame MSE can remain comparatively small even when trajectory-space metrics explode. Pixel-level success is therefore not enough.

---

## Conclusions

### Summary of Findings

This tiny-video study supports the same high-level conclusion as the broader benchmark literature:

1. **The models fit the training distribution very well.**
2. **Long context helps in-distribution rollout quality.**
3. **OOD physics remains poor for all models.**
4. **Perfect ID probes do not imply true physics understanding.**

The simplest interpretation is:

> these models learn compact representations of the training trajectories, but they do **not** learn a robust physical world model that transfers under parameter shift.

### Takeaways

1. **Visual realism ≠ physics understanding.**
2. **Context length matters, but not enough.** Longer context helps ID fit while still failing OOD.
3. **Linear probes need OOD validation.** Probe success alone can dramatically overstate “physics emergence.”

### Limitations

- Only two toy motion families were tested.
- The model class is deliberately tiny.
- Linear probes are a weak diagnostic compared with interventions or counterfactual rollouts.

### Future Work

- Add explicit state bottlenecks or dynamics heads
- Enforce stronger local-dynamics inductive bias during training
- Probe causal interventions rather than only linear decodability
- Increase dataset diversity while maintaining known ground-truth physics

---

## References

1. From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers. arXiv:2602.06923.
2. Do generative video models understand physical principles? arXiv:2501.09038.
3. Morpheus: Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments. arXiv:2504.02918.

---

## Appendix: Reproducibility

- Training entry point: `scripts/run_all.py`
- Main package: `worldmodelphy/`
- Main result summary: `artifacts/summary.json`
- Per-run outputs: `artifacts/runs/<experiment>/`

Reproduce everything with:

```bash
/data2/miniconda3/condabin/conda run -n base python scripts/run_all.py
```

That command regenerates datasets, retrains the four models, exports GIFs/figures/tables, and refreshes report assets.
