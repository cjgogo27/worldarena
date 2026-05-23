# Reviewer Supplement Manifest (Own Method Only)

- Generated: 2026-04-09T11:20:56.075879Z
- Total copied files: 21
- Text files rewritten/anonymized: 8

## Included Sections
- `code/`: own method code only (no third-party comparison adapters).
- `configs/`: method-focused configs aligned with appendix experiments.
- `datasets/`: representative + top30 skill examples and source skill updates.
- `artifacts/`: method metrics and failure-case image assets.
- `docs/`: method alignment and failure analysis notes.

## Source Mapping (sample)
- `src/skill_parser.py` -> `code/src/skill_parser.py`
- `src/skill_library.py` -> `code/src/skill_library.py`
- `src/skill_optimizer.py` -> `code/src/skill_optimizer.py`
- `src/evaluator.py` -> `code/src/evaluator.py`
- `src/vlm_client.py` -> `code/src/vlm_client.py`
- `src/web_search.py` -> `code/src/web_search.py`
- `src/GeoVista/skill_graph_runtime.py` -> `code/src/geoskill_graph_runtime.py`
- `scripts/summarize_rollout_trace.py` -> `code/scripts/summarize_rollout_trace.py`
- `configs/review_evolution_georc50_provider_a.yaml` -> `configs/config_review_evolution_ours.yaml`
- `configs/ablation_geovista_skill_graph_rollout3_georc50.yaml` -> `configs/config_ablation_rollout3_ours.yaml`
- `docs/skill_examples_appendix_selected.jsonl` -> `datasets/skill_examples_appendix_selected.jsonl`
- `docs/skill_examples_appendix_selected.csv` -> `datasets/skill_examples_appendix_selected.csv`
- `docs/skill_examples_appendix_selected.md` -> `datasets/skill_examples_appendix_selected.md`
- `docs/skill_examples_top30.jsonl` -> `datasets/skill_examples_top30.jsonl`
- `docs/skill_examples_top30.csv` -> `datasets/skill_examples_top30.csv`
- `docs/skill_examples_top30.md` -> `datasets/skill_examples_top30.md`
- `docs/skill_examples_top30_plain.md` -> `datasets/skill_examples_top30_plain.md`
- `experiments/full_100_provider_a/skill_updates/fused_geoskill_skill_graph_r1_20260401_052215.jsonl` -> `datasets/source_skill_updates/fused_geoskill_skill_graph_r1_20260401_052215.jsonl`
- `experiments/full_100_provider_a/skill_updates/recovered_geoskill_skill_graph_r1_20260401_052215.jsonl` -> `datasets/source_skill_updates/recovered_geoskill_skill_graph_r1_20260401_052215.jsonl`
- `experiments/review_evolution_georc50_provider_a/skill_updates/recovered_skill_conditioned_v3_r3_20260402_105908.jsonl` -> `datasets/source_skill_updates/recovered_skill_conditioned_v3_r3_20260402_105908.jsonl`
- ... and 1 more files
