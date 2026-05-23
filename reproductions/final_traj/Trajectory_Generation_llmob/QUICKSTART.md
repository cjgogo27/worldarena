# LLMob 快速复现指南

LLMob 已作为 FinalTraj 的独立 baseline 放在：

```bash
/data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
```

## 1. 不消耗 API 的检查

先确认代码、数据和基础依赖是否可用：

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
./run_smoke_check.sh 2019
```

如果缺依赖：

```bash
python -m pip install -r requirements.txt
```

## 2. 配置 OpenAI Key

推荐使用环境变量，不要把 key 写进仓库文件：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_MODEL="gpt-4o-mini"
```

论文原始模型是 `gpt-3.5-turbo-0613`；当前仓库默认是 `gpt-4o-mini`，结果会有差异。

## 3. 启动复现

参数说明：

- dataset：`2019`、`2021`、`20192021`
- mode：`1` = LLMob-E，`0` = LLMob-L

```bash
cd /data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob
./start_generation.sh 2019 1
```

为了先控制 API 成本，可以只跑少量用户：

```bash
./start_generation.sh 2019 1 1
```

查看状态：

```bash
screen -ls
tail -f /data/alice/cjtest/FinalTraj/logs/llmob_generate_2019_mode1_*.log
```

## 4. 手动运行命令

```bash
python generate.py --dataset 2019 --mode 1
python evaluate.py --dataset 2019 --mode 1
```

小样本试跑：

```bash
python generate.py --dataset 2019 --mode 1 --max_users 1
python evaluate.py --dataset 2019 --mode 1
```

六组论文复现命令：

```bash
python generate.py --dataset 2019 --mode 1 && python evaluate.py --dataset 2019 --mode 1
python generate.py --dataset 2019 --mode 0 && python evaluate.py --dataset 2019 --mode 0
python generate.py --dataset 2021 --mode 1 && python evaluate.py --dataset 2021 --mode 1
python generate.py --dataset 2021 --mode 0 && python evaluate.py --dataset 2021 --mode 0
python generate.py --dataset 20192021 --mode 1 && python evaluate.py --dataset 20192021 --mode 1
python generate.py --dataset 20192021 --mode 0 && python evaluate.py --dataset 20192021 --mode 0
```

## 5. 输出位置

原始 LLMob 结果会写到：

```bash
result/normal/generated/llm_e/
result/normal/generated/llm_l/
result/abnormal/generated/llm_e/
result/abnormal/generated/llm_l/
result/normal_abnormal/generated/llm_e/
result/normal_abnormal/generated/llm_l/
```

评估指标由 `evaluate.py` 输出：`SD`、`SI`、`DARD`、`STVD`，均为 JSD，越低越好。

## 6. 注意事项

- 完整复现会触发大量 OpenAI API 调用，建议先跑单个 dataset/mode。
- `LLMob-L` 会额外训练一个轻量检索模型，耗时更长。
- 该仓库使用 Foursquare pickle 数据，与 FinalTraj 的 NHTS JSON 轨迹格式不同；当前步骤是先复现 LLMob 原论文代码，再做格式适配。
