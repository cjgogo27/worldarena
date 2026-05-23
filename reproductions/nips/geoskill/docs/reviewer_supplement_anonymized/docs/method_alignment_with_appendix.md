# Method-Only Alignment Note

This package intentionally includes only our method-related implementation and artifacts.

## Included Method Code
- `code/src/geoskill_method_predictor.py`: extracted predictor core for `skill_conditioned_v3` and `skill_conditioned_v4`.
- `code/src/skill_parser.py`: skill parsing and geographic priors.
- `code/src/skill_library.py`: skill retrieval (BM25 + embedding hybrid).
- `code/src/skill_optimizer.py`: failure-recovery and skill fusion generation.
- `code/src/geoskill_graph_runtime.py`: skill-graph plan construction and rollout edge summaries.
- `code/src/evaluator.py`: evaluation metric aggregation used by our method runs.

## Naming in This Package
- The previous internal method alias is replaced by `geoskill_skill_graph` for anonymous review.
- `skill_conditioned_v3` and `skill_conditioned_v4` names are preserved where they already match appendix descriptions.

## Exclusion Policy
- No third-party comparison adapters are included in this package.
- No comparison-only runner scripts are included in this package.
