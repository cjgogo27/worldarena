# Bhandari24 快速复现指南

本文对应：**Urban Mobility Assessment Using LLMs**，Prabin Bhandari, Antonios Anastasopoulos, Dieter Pfoser，SIGSPATIAL 2024。

官方代码已拉到 FinalTraj：

```bash
/data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
```

## 1. 不消耗 API 的检查

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24
./run_smoke_check.sh sf
```

这个检查会验证：

- NHTS 2017 processed data
- Census ACS data
- training CSV
- paper pre-generated processed outputs
- CSV columns needed for后续分析

## 2. 查看 CLI

```bash
python main.py --help
```

我已经补了 `--num_samples`，默认仍是原仓库的 500 条；试跑时建议先用 1。

## 3. 安装依赖

仅做 smoke check / CLI：

```bash
python -m pip install -r requirements-smoke.txt
```

完整复现环境使用官方 pinned 依赖：

```bash
python -m pip install -r requirements.txt
```

注意：官方 `requirements.txt` 很重，包含 `torch==2.0.1`、`deepspeed`、`vllm`、`peft` 等，建议单独 conda/venv 环境安装。

## 4. 低成本生成试跑

OpenAI GPT-3.5 后端：

```bash
export OPENAI_API_KEY="sk-..."
./start_generation.sh sf gpt3 1
```

参数含义：

- `sf`：城市，可选 `sf`、`dc`、`dfw`、`minneapolis`、`la`
- `gpt3`：模型后端
- `1`：只生成 1 个样本，避免直接跑 500 条

输出位置：

```bash
outputs/<location>_<model>_<timestamp>/
```

日志位置：

```bash
/data/alice/cjtest/FinalTraj/logs/bhandari24_generate_*.log
```

## 5. 论文式复现入口

官方代码主入口：

```bash
python main.py --type completion --location sf --use_model gpt3 --year 2017 --num_samples 500 --out_folder outputs/sf_gpt3
```

其他后端：

```bash
python main.py --type completion --location sf --use_model qwen-local --qwen_model_path /data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B --year 2017 --num_samples 1 --out_folder outputs/sf_qwen_smoke
python main.py --type completion --location sf --use_model gpt4 --year 2017 --num_samples 500 --out_folder outputs/sf_gpt4
python main.py --type completion --location sf --use_model gemini --year 2017 --num_samples 500 --out_folder outputs/sf_gemini
python main.py --type completion --location sf --use_model llama2-70b --year 2017 --num_samples 500 --out_folder outputs/sf_llama2_70b
python main.py --type completion --location sf --use_model llama-2-trained --year 2017 --trained_model_epoch 3 --trained_db_size 10000-exclude-inf-cities --num_samples 500 --out_folder outputs/sf_llama2_trained
```

## 6. 后端凭证

- `gpt3`：`OPENAI_API_KEY` 或文件 `openai_key_new`
- `gpt4`：`AZURE_OPENAI_API_KEY`，可选 `AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_DEPLOYMENT`
- `gemini`：`GOOGLE_API_KEY` 或原仓库的 `palm_api_key` / `palm_api_key2`
- `llama2-70b` / `llama-2-trained`：HuggingFace Llama-2 权限 token，且需要足够 GPU 资源

## 7. 预生成结果

仓库自带 `outputs_processed/`，可先用来复核论文输出格式和分析脚本，不需要重新调用 LLM。

## 8. 与 FinalTraj 的关系

Bhandari24 输出是 travel survey / activity diary 表格，核心列为：

- `place_name`
- `arrival_time`
- `departure_time`
- `loc_type`

FinalTraj 中央 evaluator 需要 `user_id + schedule(activity/start_time/end_time)` JSON。下一步如需纳入统一评估，需要增加一个 converter，把 Bhandari24 的 `loc_type/place_name` 映射到 FinalTraj 的 10 类 activity taxonomy。
