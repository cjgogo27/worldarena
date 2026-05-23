# LaRA-WM Results Summary

## Valid Offline Baselines

### Internal baselines (full held-out RoboTwin test split)

| Model | Action MSE | Action MAE | Action R2 |
| --- | ---: | ---: | ---: |
| Direct Policy | 0.1961 ± 0.0029 | 0.2520 ± 0.0064 | 0.2292 ± 0.0116 |
| Latent No-Refine | 0.1933 ± 0.0026 | 0.2518 ± 0.0015 | 0.2400 ± 0.0103 |
| No-Reward WM | 0.1937 ± 0.0044 | 0.2501 ± 0.0012 | 0.2385 ± 0.0173 |
| LaRA-WM | 0.1785 ± 0.0053 | 0.2391 ± 0.0065 | 0.2983 ± 0.0207 |

### External baselines (offline, same fair metrics)

| Model | Action MSE | Action MAE | Action R2 | Notes |
| --- | ---: | ---: | ---: | --- |
| ACT | 0.0083 | 0.0405 | 0.9533 | Full offline reproduction completed |
| Diffusion Policy | 0.0077 | 0.0541 | 0.9535 | Full offline reproduction completed |
| UniVLA + latent decoder | 0.0116 | 0.0803 | 0.9344 | Public qwbu/univla-7b backbone + RoboTwin decoder |
| OpenVLA (3 ep probe) | 0.8041 | 0.7129 | 0.6464 | Zero-shot probe; smaller evaluation subset |
| UniVLA direct-action (3 ep probe) | 2.4561 | 1.3092 | very negative | Zero-shot direct-action probe is not a strong final baseline |

## Success-rate Evaluation Status

The RoboTwin-native success evaluation entrypoint is implemented in:

- `scripts/run_robotwin_success_eval.py`

and currently supports rollout wrappers for:

- `lara_wm`
- `direct_policy`
- `latent_no_refine`

### Smoke results already measured

Using the native RoboTwin task-success predicate on `grab_roller` with `demo_clean`, unseen instructions, and `test_num=1`:

| Model | SR |
| --- | ---: |
| LaRA-WM | 0.00 |
| Direct Policy | 0.00 |
| Latent No-Refine | 0.00 |

These runs confirm that the success-based metric pipeline is now executing end-to-end. However, they are only smoke-test results; final paper-safe SR/ID/OOD tables still require a broader sweep across tasks/configs and the remaining runtime cleanup for more scalable rollout evaluation.

## Figure Assets

Prepared figure assets/snippets:

- `src/reports/figure_snippets.tex`
- `src/reports/figures/grab_roller/frame_01.png`
- `src/reports/figures/grab_roller/frame_02.png`
- `src/reports/figures/grab_roller/frame_03.png`

## LaTeX Tables

- `src/reports/table3_expanded_offline_comparison.tex`
- `src/reports/table4_zero_shot_vla_probe.tex`
- `src/reports/table5_success_metric_protocol.tex`
- `src/reports/table6_success_smoke.tex`
