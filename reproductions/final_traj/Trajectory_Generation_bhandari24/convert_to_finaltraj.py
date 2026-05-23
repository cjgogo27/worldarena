"""Convert Bhandari24 processed CSV to FinalTraj evaluation JSON schedule format.

Usage:
    python convert_to_finaltraj.py \\
        --csv outputs_processed_qwen/outputs_processed_completion_sf_qwen_smoke.csv \\
        --json output_trajectories/finaltraj_bhandari24_sf_qwen.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


NHTS_TO_FINALTRAJ = {
    1: "home",
    2: "home",
    3: "work",
    4: "work",
    5: "service",
    6: "dropoff_pickup",
    7: "service",
    8: "education",
    9: "education",
    10: "service",
    11: "shopping",
    12: "service",
    13: "dine_out",
    14: "service",
    15: "socialize",
    16: "exercise",
    17: "socialize",
    18: "medical",
    19: "socialize",
    97: "service",
}


def parse_time(tstr: str) -> str:
    """Convert 'hh:mm AM/PM' or 'HHMM' to 'HH:MM' 24h format."""
    tstr = tstr.strip()
    m = re.match(r"(\d{1,2}):(\d{2})\s*([AP]M)", tstr, re.I)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ampm = m.group(3).upper()
        if ampm == "PM" and hh != 12:
            hh += 12
        if ampm == "AM" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"
    m = re.match(r"(\d{3,4})", tstr)
    if m:
        s = m.group(1).zfill(4)
        return f"{s[:2]}:{s[2:]}"
    return tstr


def convert_row_per_visit_csv(csv_path: str, uuid_prefix: str = "bhandari24") -> list[dict]:
    """Convert Bhandari24's row-per-visit CSV to FinalTraj JSON format."""
    users: dict[str, list[dict]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uuid", "")
            if not uid:
                continue

            loc_type_raw = row.get("loc_type", "97").strip()
            try:
                loc_type = int(re.match(r"(\d+)", loc_type_raw).group(1))
            except (AttributeError, ValueError):
                loc_type = 97

            activity = NHTS_TO_FINALTRAJ.get(loc_type, "service")
            arrival = parse_time(row.get("arrival_time", "00:00"))
            departure = parse_time(row.get("departure_time", "00:00"))

            if uid not in users:
                users[uid] = []
            users[uid].append({
                "activity": activity,
                "start_time": arrival,
                "end_time": departure,
            })

    result = []
    for uid, schedule in users.items():
        schedule.sort(key=lambda s: s["start_time"])
        result.append({
            "user_id": f"{uuid_prefix}_{uid}",
            "schedule": schedule,
        })

    return result


def convert_aggregated_csv(csv_path: str, uuid_prefix: str = "bhandari24") -> list[dict]:
    """Convert aggregated (one-row-per-user) CSV to FinalTraj JSON format.

    This handles the format where loc_type_gen and stay_time_gen are
    stored as stringified JSON arrays in a single row per user.
    """
    import ast
    result = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uuid", "") or row.get("key", "")
            if not uid:
                continue

            loc_type_str = row.get("loc_type_gen", "[]")
            stay_time_str = row.get("stay_time_gen", "[]")
            stay_times_str = row.get("stay_times", "[]")

            try:
                loc_types = ast.literal_eval(loc_type_str)
            except Exception:
                loc_types = []
            try:
                stay_times = ast.literal_eval(stay_time_str)
            except Exception:
                stay_times = []
            if not stay_times:
                try:
                    stay_times = ast.literal_eval(stay_times_str)
                except Exception:
                    stay_times = []

            if not loc_types or not stay_times:
                continue

            schedule = []
            for i in range(min(len(loc_types), len(stay_times))):
                code = loc_types[i]
                activity = NHTS_TO_FINALTRAJ.get(int(code), "service")
                start_raw = str(stay_times[i][0]).zfill(4)
                end_raw = str(stay_times[i][1]).zfill(4)
                start_time = f"{start_raw[:2]}:{start_raw[2:]}"
                end_time = f"{end_raw[:2]}:{end_raw[2:]}"
                if end_time == "24:00":
                    end_time = "23:59"
                schedule.append({
                    "activity": activity,
                    "start_time": start_time,
                    "end_time": end_time,
                })

            result.append({
                "user_id": f"{uuid_prefix}_{uid}",
                "schedule": schedule,
            })

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bhandari24 CSV to FinalTraj JSON schedule format."
    )
    parser.add_argument("--csv", required=True, help="Path to processed CSV")
    parser.add_argument("--json", required=True, help="Output JSON path")
    parser.add_argument(
        "--prefix", default="bhandari24",
        help="Prefix for user_id in output (default: bhandari24)"
    )
    parser.add_argument(
        "--format", choices=["auto", "row-per-visit", "aggregated"],
        default="auto",
        help="CSV format (auto-detect by checking for loc_type_gen column)"
    )
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    if args.format == "row-per-visit" or (args.format == "auto" and "loc_type_gen" not in header):
        data = convert_row_per_visit_csv(args.csv, args.prefix)
    else:
        data = convert_aggregated_csv(args.csv, args.prefix)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Converted {len(data)} users to {args.json}")


if __name__ == "__main__":
    main()
