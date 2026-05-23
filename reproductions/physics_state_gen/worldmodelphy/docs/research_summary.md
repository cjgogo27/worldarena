# Research Summary

## Main synthesis

Current evidence does **not** support the claim that mainstream video generative models have robustly learned physics. The dominant pattern in recent literature is:

- visually plausible motion is common
- causal, mechanistic, and OOD-robust physics is still weak
- benchmark success is much lower than subjective visual realism suggests

## Key references synthesized

1. `From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers` (arXiv:2602.06923)
   - prediction accuracy alone does not imply physics understanding
   - temporal locality is crucial for shifting from global trajectory fitting to local dynamics representations
2. `Do generative video models understand physical principles?` / Physics-IQ (arXiv:2501.09038)
   - strong visual realism can coexist with weak physical understanding
3. `Morpheus: Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments` (arXiv:2504.02918)
   - current models struggle with conservation-law-grounded physical plausibility

## Working hypothesis for this repository

For very small models trained from scratch on synthetic videos:

- short-context models may better encode local dynamics
- long-context models may overfit to trajectory templates
- OOD generalization and latent linear probes are the critical tests
