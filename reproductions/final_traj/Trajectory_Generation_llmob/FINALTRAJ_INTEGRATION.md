# LLMob in FinalTraj

This directory is a direct copy of the official LLMob implementation from:

https://github.com/Wangjw6/LLMob

It is kept as a standalone FinalTraj baseline, matching the project convention used by `Trajectory_Generation_tradition/`, `Trajectory_Generation_tradition2/`, and `CoPB/`.

## Reproduction scope

The first reproducible target is the original LLMob paper setup using the included Foursquare pickle data:

- `2019`, normal scenario
- `2021`, abnormal scenario
- `20192021`, cross-period scenario
- `mode=1`, LLMob-E
- `mode=0`, LLMob-L

## Expected paper metrics

The paper reports four Jensen-Shannon divergence metrics, lower is better:

| Scenario | Model | SD | SI | DARD | STVD |
| --- | --- | ---: | ---: | ---: | ---: |
| 2019 normal | LLMob-E | 0.053 | 0.046 | 0.125 | 0.559 |
| 2019 normal | LLMob-L | 0.049 | 0.054 | 0.136 | 0.570 |
| 2021 abnormal | LLMob-E | 0.056 | 0.043 | 0.127 | 0.615 |
| 2021 abnormal | LLMob-L | 0.057 | 0.051 | 0.124 | 0.609 |
| 2019 to 2021 | LLMob-E | 0.062 | 0.056 | 0.117 | 0.536 |
| 2019 to 2021 | LLMob-L | 0.064 | 0.051 | 0.124 | 0.531 |

Exact numbers may differ because OpenAI models have changed since the NeurIPS 2024 paper. The repository defaults to `gpt-4o-mini`; the paper used `gpt-3.5-turbo-0613`.

## FinalTraj adaptation notes

LLMob's native output is stored as pickle files under `result/.../generated/.../<user_id>/results.pkl`. FinalTraj's central evaluator expects JSON schedules shaped like:

```json
[
  {
    "user_id": "30007884_1",
    "schedule": [
      {"activity": "home", "start_time": "00:00", "end_time": "07:30"}
    ]
  }
]
```

So the next integration step after paper reproduction is a converter from LLMob pickle trajectories into FinalTraj JSON schedules. That converter should map Foursquare categories into FinalTraj's activity taxonomy:

- `home`
- `work`
- `education`
- `shopping`
- `service`
- `medical`
- `dine_out`
- `socialize`
- `exercise`
- `dropoff_pickup`

Until that converter is added, use LLMob's own `evaluate.py` for paper reproduction metrics.
