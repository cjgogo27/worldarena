# LLMob Reproduction Status

Date: 2026-05-19

## Completed

- Cloned the official LLMob repository into `FinalTraj/Trajectory_Generation_llmob/`.
- Added FinalTraj-style helper files:
  - `QUICKSTART.md`
  - `FINALTRAJ_INTEGRATION.md`
  - `requirements.txt`
  - `smoke_check.py`
  - `run_smoke_check.sh`
  - `start_generation.sh`
  - `output_trajectories/.gitkeep`
- Added optional `generate.py` controls for low-cost reproduction:
  - `--max_users N`
  - `--user_ids 13,1004`
- Fixed safe configuration behavior:
  - environment variables override blank YAML values
  - `OPENAI_API_KEY` is no longer printed
  - `generate.py --help` works without an API key
- Fixed one typo in the OpenAI rate limiter: `elapsed_tim` -> `elapsed_time`.

## Verification run

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
python -m py_compile generate.py evaluate.py smoke_check.py engine/llm_configs/config.py engine/llm_configs/openai_api.py
./run_smoke_check.sh 2019
python generate.py --help
env -u OPENAI_API_KEY ./start_generation.sh 2019 1 1
```

Observed smoke-check result:

```text
LLMob smoke check
  project: /data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
  dataset: 2019
  users: 93
  sample_user: 1004
  sample_pickle_items: 12
  train_days: 57
  test_days: 15
  loc_map_entries: 248555
  pos_map_entries: 248555
  activity_map_entries: 867
All no-cost checks passed. Generation still requires OPENAI_API_KEY.
```

## Remaining blocker

Full LLMob reproduction requires an OpenAI API key and will trigger many API calls. The safe first paid run is:

```bash
export OPENAI_API_KEY="sk-..."
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
./start_generation.sh 2019 1 1
```

After that succeeds, run the full paper commands from `QUICKSTART.md`.

## Known upstream typing status

The copied upstream LLMob code has existing static-type diagnostics under the local basedpyright configuration, mostly from old OpenAI SDK attributes and untyped research code. Syntax compilation and no-cost data validation pass. The next runtime-risk area, once an API key is available, is the mixed old/new OpenAI SDK usage in upstream helper functions; the main generation path already uses `OpenAI().chat.completions.create` in `engine/llm_configs/gpt_structure.py`.
