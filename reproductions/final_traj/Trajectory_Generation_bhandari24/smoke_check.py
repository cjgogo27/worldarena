"""No-cost checks for the Bhandari24 Urban-Mobility-LLM baseline.

This validates repository data and pre-generated outputs without calling any
LLM API and without loading Llama-2 weights.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Bhandari24 data and outputs without LLM calls.")
    parser.add_argument("--location", default="sf", choices=["sf", "dc", "dfw", "minneapolis", "la"])
    parser.add_argument("--model", default="llama", choices=["llama", "gpt4", "gemini", "llama_trained"])
    args = parser.parse_args()

    required_paths = [
        PROJECT_DIR / "main.py",
        PROJECT_DIR / "model_inference.py",
        PROJECT_DIR / "generate_system_prompt.py",
        PROJECT_DIR / "sample_population.py",
        PROJECT_DIR / "dataset" / "trip_ids.csv",
        PROJECT_DIR / "dataset" / "NHTS_2017_csv" / "processed_data" / args.location,
        PROJECT_DIR / "dataset" / "census_data",
        PROJECT_DIR / "training_datasets" / "training_dataset_small.csv",
        PROJECT_DIR / "outputs_processed",
    ]
    missing = [str(path.relative_to(PROJECT_DIR)) for path in required_paths if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    output_candidates = sorted((PROJECT_DIR / "outputs_processed").glob(f"outputs_processed_completion_{args.location}_*.csv"))
    if not output_candidates:
        print(f"No processed outputs found for location={args.location}")
        return 1

    training_header = read_header(PROJECT_DIR / "training_datasets" / "training_dataset_small.csv")
    expected_training_columns = {"context", "output"}
    if not expected_training_columns.issubset(set(training_header)):
        print(f"Training dataset missing columns: {sorted(expected_training_columns - set(training_header))}")
        return 1

    output_path = output_candidates[0]
    output_header = read_header(output_path)
    expected_output_columns = {"sex", "age", "location", "survey_date", "place_name", "arrival_time", "departure_time", "loc_type"}
    if not expected_output_columns.issubset(set(output_header)):
        print(f"Processed output missing columns: {sorted(expected_output_columns - set(output_header))}")
        return 1

    nhts_path = PROJECT_DIR / "dataset" / "NHTS_2017_csv" / "processed_data" / args.location
    if nhts_path.is_dir():
        nhts_summary = f"{len(list(nhts_path.iterdir()))} entries"
    else:
        nhts_summary = f"file, {nhts_path.stat().st_size} bytes"
    census_files = list((PROJECT_DIR / "dataset" / "census_data").glob("**/*.csv"))
    training_rows = count_csv_rows(PROJECT_DIR / "training_datasets" / "training_dataset_small.csv")
    output_rows = count_csv_rows(output_path)

    print("Bhandari24 smoke check")
    print(f"  project: {PROJECT_DIR}")
    print(f"  location: {args.location}")
    print(f"  nhts_processed_data: {nhts_summary}")
    print(f"  census_csv_files: {len(census_files)}")
    print(f"  training_small_rows: {training_rows}")
    print(f"  processed_output_file: {output_path.name}")
    print(f"  processed_output_rows: {output_rows}")
    print(f"  processed_output_variants_for_location: {len(output_candidates)}")
    print("All no-cost checks passed. New generation requires an API key or Llama-2/HF GPU setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
