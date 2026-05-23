# Simulation Review and Fixes (2026-03-28)

This note records a manual protocol simulation and the issues found.

## 1) Simulation Setup (Manual Dry-Run)
- Simulated one condition group with K=6 candidates.
- Used ranking-based pair construction with 1:3 policy.
- Checked expansion to DPO pairs and gate checks.

## 2) Simulated Cases and Outcomes

Case A: Normal score spread
- Scores: [0.83, 0.79, 0.72, 0.61, 0.56, 0.49]
- chosen=0.83, negatives={0.49,0.56,0.61}
- Result: pass, expands to 3 DPO pairs with pair_weight=1/3 each.

Case B: Near-flat scores
- Scores: [0.70001, 0.70000, 0.70000, 0.69999]
- Risk: pseudo-random pairing due to negligible margin.
- Fix: add uninformative-group filter: if max-min < 1e-6, drop group.

Case C: Candidate shortage after filtering
- Valid candidates reduced to 3 due to corrupt image + dedup.
- Result: fallback to 1:2.
- Fix: mandatory fallback logging and ratio distribution reporting.

Case D: VLM unavailable
- Risk: silent CLIP-only fallback corrupts benchmark definition.
- Fix: disallow silent fallback; mark run invalid for final benchmark.

Case E: Split leakage risk
- Risk: same content_id appears in train and val.
- Fix: split by content_id before pairing; add leakage gate.

## 3) Issues Identified
1. Missing deterministic scorer config caused reproducibility risk.
2. Missing failure artifact path made debugging hard.
3. Missing drop-rate threshold could allow low-quality benchmark builds.
4. Missing VLM/CLIP disagreement diagnosis could hide scoring drift.

## 4) Fixes Applied to Protocol
- Added deterministic scorer requirements (temperature=0, version logging).
- Added `pair_build_failures.jsonl` as mandatory output.
- Added uninformative-group drop rule (max-min < 1e-6).
- Added dropped-group ratio hard gate (<=30%).
- Added VLM failure and VLM/CLIP disagreement handling.

## 5) Final Verdict
The revised protocol is now materially safer for real execution. Remaining uncertainty is operational (compute budget and scorer API stability), not logic-level ambiguity.
