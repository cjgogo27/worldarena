#!/usr/bin/env python3

import json
import os
import sys
import re
import random
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_system_prompt import generate_completion_prompt
from model_inference import conduct_qwen_local_completion
from transformers import AutoTokenizer, AutoModelForCausalLM

BHANDARI_DIR = Path(__file__).parent
CA_SELECTION = Path("/data/alice/cjtest/FinalTraj/ca_selection_10.json")
CA_PERSON_STATIC = Path("/data/alice/cjtest/FinalTraj/California/data_person/california_person_static.json")
GT_JSON = Path("/data/alice/cjtest/FinalTraj/California/processed_data/all_user_schedules.json")
QWEN_PATH = "/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
OUT_DIR = BHANDARI_DIR / "outputs_ca"
OUT_FINALTRAJ = BHANDARI_DIR / "output_trajectories" / "finaltraj_bhandari24_ca.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CA_SELECTION) as f:
    selection = json.load(f)

with open(CA_PERSON_STATIC) as f:
    person_static = json.load(f)
ps_by_id = {p["user_id"]: p for p in person_static}

with open(GT_JSON) as f:
    gt_data = json.load(f)
gt_by_id = {u["user_id"]: u["schedule"] for u in gt_data}

user_ids = selection["user_ids"]
print(f"Selected {len(user_ids)} CA users: {user_ids}")

AGE_ESTIMATE = {
    "0-4 years old": 3,
    "5-15 years old": 10,
    "16-17 years old": 16,
    "18-64 years old": 40,
    "65-75 years old": 70,
    "76 years old or older": 80,
}

def infer_age(age_range: str, primary_activity: str) -> int:
    if age_range in AGE_ESTIMATE:
        return AGE_ESTIMATE[age_range]
    pa = primary_activity.lower()
    if "retire" in pa:
        return 70
    if "school" in pa or "student" in pa:
        return 17
    if "homemaker" in pa:
        return 35
    return 40

def map_primary_activity(pa: str):
    pa_lower = pa.lower()
    if "school" in pa_lower:
        return True, False, False
    if "working" in pa_lower:
        return False, True, True
    if "looking for work" in pa_lower or "unemployed" in pa_lower:
        return False, True, False
    if "retire" in pa_lower or "homemaker" in pa_lower or "something else" in pa_lower:
        return False, False, False
    return False, False, False

def map_marital_status(relationship: str) -> str:
    r = relationship.lower()
    if "spouse" in r or "partner" in r:
        return "married"
    if "self" in r:
        return "never married"
    if "child" in r:
        return "never married"
    if "parent" in r:
        return "never married"
    return "never married"

def map_survey_date(gt_schedule: list) -> tuple:
    """Grab a reasonable survey date from ground truth or default."""
    # Default: a Tuesday in April 2017
    return date(2017, 4, 25), "Tuesday"


# NHTS location type → FinalTraj activity mapping (same as convert_to_finaltraj.py)
NHTS_TO_FINALTRAJ = {
    1: "home", 2: "home", 3: "work", 4: "work", 5: "service",
    6: "dropoff_pickup", 7: "service", 8: "education", 9: "education",
    10: "service", 11: "shopping", 12: "service", 13: "dine_out",
    14: "service", 15: "socialize", 16: "exercise", 17: "socialize",
    18: "medical", 19: "socialize", 97: "service",
}


def parse_bhandari24_table(text: str) -> list[dict]:
    """Parse markdown table from Bhandari24 LLM output into schedule entries."""
    entries = []
    lines = text.strip().split("\n")
    in_table = False
    for line in lines:
        # Detect table row: starts with |
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().split("|")]
            # Filter out header/separator rows
            if len(cells) >= 5:
                place = cells[1]
                arrival = cells[2]
                departure = cells[3]
                loc_type_raw = cells[4]
                # Skip header rows
                if place.lower() in ("place visited", "place name", ""):
                    continue
                if all(c == "-" or c == "" for c in place):
                    continue
                m = re.search(r"(\d+)", loc_type_raw)
                loc_code = int(m.group(1)) if m else 97
                activity = NHTS_TO_FINALTRAJ.get(loc_code, "service")
                entry = {
                    "activity": activity,
                    "start_time": convert_time(arrival),
                    "end_time": convert_time(departure),
                }
                # Skip obviously invalid entries
                if entry["start_time"] and entry["end_time"]:
                    entries.append(entry)
    return entries


def convert_time(tstr: str) -> str:
    tstr = tstr.strip()
    m = re.match(r"(\d{1,2}):(\d{2})\s*([AP]M)", tstr, re.I)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ap = m.group(3).upper()
        if ap == "PM" and hh != 12:
            hh += 12
        if ap == "AM" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"
    m = re.match(r"(\d{1,2}):(\d{2})", tstr)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return "00:00"


def location_for_user(uid: str) -> str:
    """Use appropriate CA metro area location string."""
    return "San Francisco-Oakland-Hayward, CA Metro Area"


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("Loading Qwen3-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    survey_date, survey_weekday = date(2017, 4, 25), "Tuesday"

    all_outputs = []

    for uid in user_ids:
        p = ps_by_id.get(uid)
        if p is None:
            print(f"  Skipping {uid}: no demographics")
            continue

        gt_schedule = gt_by_id.get(uid, [])

        # Map demographics
        sex = p["gender"].lower()  # "male" / "female"
        age_range = p["age_range"]
        age = infer_age(age_range, p.get("primary_activity", ""))
        race = p["race"]
        enrolled, labor_force, employed = map_primary_activity(p.get("primary_activity", ""))

        occupation = p.get("occupation", "None")
        if occupation in ("Appropriate skip", "None", "", "Not specified"):
            occupation = "None"

        marital = map_marital_status(p.get("relationship", ""))
        household_type = "non family household" if marital == "never married" else "married couple family"
        own_child = None
        own_child_type = None
        location = location_for_user(uid)
        loc_ub = "none"

        if age < 16:
            print(f"  Skipping {uid}: age {age} < 16")
            continue

        # Generate prompt
        prompt = generate_completion_prompt(
            sex, age_range, age, race,
            enrolled, labor_force, employed,
            occupation, marital, household_type,
            own_child, own_child_type,
            location, survey_date, survey_weekday, loc_ub,
        )

        print(f"\nGenerating for {uid} ({sex}, {age}, {p.get('primary_activity','?')})...")

        # Run inference
        try:
            result = conduct_qwen_local_completion(model, tokenizer, prompt_file=prompt)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        output_text = result.get("ans_output", "")

        # Parse table from output
        schedule = parse_bhandari24_table(output_text)

        entry = {
            "user_id": uid,
            "input_demographics": {
                "sex": sex, "age": age, "age_range": age_range, "race": race,
                "enrolled_in_school": enrolled, "employed": employed,
                "occupation": occupation, "marital_status": marital,
                "primary_activity": p.get("primary_activity", ""),
            },
            "raw_output": output_text,
            "schedule": schedule,
            "ground_truth": gt_schedule,
        }

        # Save individual result
        out_path = OUT_DIR / f"{uid}.json"
        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"  Saved {out_path} ({len(schedule)} activities)")

        all_outputs.append(entry)

    # ── Convert to FinalTraj JSON ──────────────────────────────────────
    finaltraj_output = []
    for entry in all_outputs:
        if entry["schedule"]:
            finaltraj_output.append({
                "user_id": entry["user_id"],
                "schedule": entry["schedule"],
            })

    OUT_FINALTRAJ.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FINALTRAJ, "w") as f:
        json.dump(finaltraj_output, f, indent=2)
    print(f"\nFinalTraj JSON saved: {OUT_FINALTRAJ} ({len(finaltraj_output)} users)")


if __name__ == "__main__":
    main()
