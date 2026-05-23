# Bhandari24 Reproduction Status

Date: 2026-05-19

## Completed

- Identified `Bhandari24` as **Urban Mobility Assessment Using LLMs**.
- Cloned official code into `FinalTraj/Trajectory_Generation_bhandari24/`.
- Added FinalTraj-style helper files:
  - `QUICKSTART.md`
  - `FINALTRAJ_INTEGRATION.md`
  - `REPRODUCTION_STATUS.md`
  - `smoke_check.py`
  - `run_smoke_check.sh`
  - `start_generation.sh`
  - `requirements-smoke.txt`
  - `output_trajectories/.gitkeep`
- Added `--num_samples` to `main.py` for low-cost reproduction.
- Made optional model backend dependencies friendlier for CLI/smoke checks.
- Added env-var credential fallback instead of only hardcoded key files.
- Added `qwen-local` backend for the local Qwen3-8B model at `/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B`.

## Verification commands

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
python main.py --help
python -m py_compile main.py model_inference.py smoke_check.py utils.py
./run_smoke_check.sh sf
env -u OPENAI_API_KEY ./start_generation.sh sf gpt3 1
```

## Local Qwen run completed

The project has been run end-to-end with the local Qwen3-8B model.

Model path:

```bash
/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B
```

Command:

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
CUDA_VISIBLE_DEVICES=5 python main.py --type completion --location sf --use_model qwen-local --year 2017 --num_samples 1 --out_folder outputs/sf_qwen_smoke
python process_output.py --type completion --in_folder outputs/sf_qwen_smoke/ --out_folder outputs_processed_qwen --file_name outputs_processed_completion_sf_qwen_smoke
```

Generated files:

```bash
outputs/sf_qwen_smoke/completions_1c6887e6-1880-4a2e-a3f9-198a6ca62469.json
outputs_processed_qwen/outputs_processed_completion_sf_qwen_smoke.csv
```

Post-processing result:

```text
Complete generations: 1
Incomplete generations: 0
Table not found: 0
```

Implementation note: Qwen3 defaults to emitting `<think>` reasoning. The local backend disables thinking via `apply_chat_template(..., enable_thinking=False)` when available and adds a table-only system instruction so Bhandari24's parser can consume the output.

## Full generation blocker

Actual new sample generation requires one of:

- `OPENAI_API_KEY` for `--use_model gpt3`
- Azure OpenAI settings for `--use_model gpt4`
- `GOOGLE_API_KEY` plus `google-generativeai` for `--use_model gemini`
- HuggingFace Llama-2 token and large GPU resources for `llama2-70b` / `llama-2-trained`

Safe first paid run:

```bash
export OPENAI_API_KEY="sk-..."
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
./start_generation.sh sf gpt3 1
```

Safe local Qwen run:

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
python main.py --type completion --location sf --use_model qwen-local --year 2017 --num_samples 1 --out_folder outputs/sf_qwen_smoke
```

## Pre-generated output status

The official repository already includes processed output CSVs under `outputs_processed/`, so paper-output format can be inspected and analysis can be developed without calling any LLM.

## Remaining FinalTraj integration work

Add a converter from Bhandari24 processed CSV rows to FinalTraj JSON schedules if this baseline needs to be scored by `FinalTraj/evaluation/evaluate_generated_trajectories.py`.
