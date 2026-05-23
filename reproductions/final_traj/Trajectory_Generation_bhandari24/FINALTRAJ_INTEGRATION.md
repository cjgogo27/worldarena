# Bhandari24 in FinalTraj

This directory contains the official code for:

**Urban Mobility Assessment Using LLMs**  
Prabin Bhandari, Antonios Anastasopoulos, Dieter Pfoser  
ACM SIGSPATIAL 2024

- Paper: https://arxiv.org/abs/2409.00063
- Official repository: https://github.com/gmuggs/Urban-Mobility-LLM
- Fine-tuned LoRA model: https://huggingface.co/prb977/Llama-2-urban-mobility
- DOI: https://doi.org/10.1145/3678717.3691221

## Why this is Bhandari24

Downstream mobility-generation papers refer to `Bhandari24` as a spatially augmented LLM mobility generation / assessment baseline. The matching paper is unambiguously `Urban Mobility Assessment Using LLMs`.

## What the code generates

The method generates synthetic travel diary tables from demographic and location prompts. The processed outputs contain rows like:

```text
sex, age, location, survey_date, place_name, arrival_time, departure_time, loc_type
```

This is not GPS trajectory generation. It is travel-survey/activity-chain generation based on ACS demographic sampling and NHTS-style diary structure.

## Data included by the repository

- `dataset/NHTS_2017_csv/processed_data/`: processed NHTS 2017 data by city
- `dataset/census_data/`: ACS census files by city/year
- `training_datasets/`: CSVs for LoRA fine-tuning data
- `outputs_processed/`: pre-generated processed outputs from multiple model backends

## FinalTraj baseline status

The baseline is integrated as a standalone directory, matching the pattern used by:

- `Trajectory_Generation_llmob/`
- `Trajectory_Generation_tradition/`
- `Trajectory_Generation_tradition2/`
- `CoPB/`

## Local changes made for reproducibility

- Added `--num_samples` to `main.py` so reproduction can start with 1 sample instead of the original fixed 500.
- Made Gemini, Replicate, vLLM, and PEFT imports optional enough for `python main.py --help` to work without installing every backend.
- Added environment-variable credential support for OpenAI, Azure OpenAI, Gemini, and Replicate.
- Added FinalTraj helper scripts and documentation.

## Unified FinalTraj evaluation gap

FinalTraj expects JSON schedules:

```json
[
  {
    "user_id": "example_1",
    "schedule": [
      {"activity": "home", "start_time": "00:00", "end_time": "08:00"}
    ]
  }
]
```

Bhandari24 processed outputs are CSV activity diary rows. To compare inside FinalTraj's central evaluator, add a converter from Bhandari24 rows to FinalTraj schedules and map `loc_type` / `place_name` into the 10 FinalTraj activity categories.

## Citation

```bibtex
@inproceedings{bhandari2024urban,
  author = {Bhandari, Prabin and Anastasopoulos, Antonios and Pfoser, Dieter},
  title = {Urban Mobility Assessment Using LLMs},
  year = {2024},
  publisher = {ACM},
  doi = {10.1145/3678717.3691221},
  booktitle = {SIGSPATIAL '24},
  pages = {67--79}
}
```
