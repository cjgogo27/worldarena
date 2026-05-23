# worldmodelphy

Tiny video generative model experiments for testing whether simple synthetic motion videos induce physics-like representations or only trajectory template memorization.

## Scope

- synthetic circular motion and projectile motion datasets
- lightweight autoregressive video generator trained from scratch
- evaluation on interpolation, OOD generalization, and linear probes for latent dynamics
- blog-style report with figures, GIFs, and generated samples

## Environment

This project is executed with:

```bash
/data2/miniconda3/condabin/conda run -n base python ...
```

## Entry Point

```bash
/data2/miniconda3/condabin/conda run -n base python scripts/run_all.py
```
