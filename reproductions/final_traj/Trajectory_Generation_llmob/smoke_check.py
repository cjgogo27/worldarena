"""No-cost sanity checks for the LLMob baseline inside FinalTraj.

This script deliberately avoids importing the OpenAI-backed generation entrypoint,
so it can run without an API key and without spending tokens.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def check_module(name: str) -> tuple[str, bool]:
    return name, importlib.util.find_spec(name) is not None


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LLMob files and local data without API calls.")
    parser.add_argument("--dataset", default="2019", choices=["2019", "2021", "20192021"])
    parser.add_argument("--sample-user", default=None, help="Optional user id, e.g. 13")
    args = parser.parse_args()

    required_paths = [
        PROJECT_DIR / "generate.py",
        PROJECT_DIR / "evaluate.py",
        PROJECT_DIR / "config" / "config.yaml",
        PROJECT_DIR / "config" / "key.yaml",
        PROJECT_DIR / "data" / "loc_map.pkl",
        PROJECT_DIR / "data" / "pos_map.pkl",
        PROJECT_DIR / "data" / "location_activity_map.pkl",
        PROJECT_DIR / "data" / args.dataset,
    ]
    missing = [str(path.relative_to(PROJECT_DIR)) for path in required_paths if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    modules = ["openai", "numpy", "scipy", "sklearn", "torch", "yaml", "geopy"]
    module_status = [check_module(name) for name in modules]
    unavailable = [name for name, ok in module_status if not ok]

    dataset_dir = PROJECT_DIR / "data" / args.dataset
    users = sorted(path.stem for path in dataset_dir.glob("*.pkl"))
    sample_user = args.sample_user or users[0]
    sample_path = dataset_dir / f"{sample_user}.pkl"
    if not sample_path.exists():
        print(f"Sample user file does not exist: {sample_path}")
        return 1

    sample = load_pickle(sample_path)
    loc_map = load_pickle(PROJECT_DIR / "data" / "loc_map.pkl")
    pos_map = load_pickle(PROJECT_DIR / "data" / "pos_map.pkl")
    activity_map = load_pickle(PROJECT_DIR / "data" / "location_activity_map.pkl")

    print("LLMob smoke check")
    print(f"  project: {PROJECT_DIR}")
    print(f"  dataset: {args.dataset}")
    print(f"  users: {len(users)}")
    print(f"  sample_user: {sample_user}")
    print(f"  sample_pickle_items: {len(sample)}")
    print(f"  train_days: {len(sample[0]) if len(sample) > 0 else 0}")
    print(f"  test_days: {len(sample[1]) if len(sample) > 1 else 0}")
    print(f"  loc_map_entries: {len(loc_map)}")
    print(f"  pos_map_entries: {len(pos_map)}")
    print(f"  activity_map_entries: {len(activity_map)}")

    if unavailable:
        print("Missing Python dependencies:")
        for name in unavailable:
            print(f"  - {name}")
        print("Install with: python -m pip install -r requirements.txt")
        return 2

    print("All no-cost checks passed. Generation still requires OPENAI_API_KEY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
