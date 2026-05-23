#!/usr/bin/env python3
"""
Geocode prediction results with GeoNames searchJSON API.

Features:
- Multiple username rotation
- Per-username hourly quota control (default 1000/hour)
- Auto switch username when API returns:
  {"status":{"message":"user does not exist.","value":10}}
- Resume support
"""

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


DEFAULT_USERNAMES = [
    "yjiang194",
    "arandinglv",
    "ouyangxin",
    "yiqiuliu",
    "ouyang_xin",
    "yiqiuliu2", 
    "yiqiuliu3", 
]


def parse_args():
    parser = argparse.ArgumentParser(description="Geocode prediction JSON with GeoNames")
    parser.add_argument("--pred_path", type=str, required=True, help="Prediction JSON path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--query_mode",
        type=str,
        default="raw",
        choices=["raw", "city_country"],
        help="raw: query by full prediction text; city_country: parse city/country first.",
    )
    parser.add_argument(
        "--usernames",
        type=str,
        default=",".join(DEFAULT_USERNAMES),
        help="Comma-separated GeoNames usernames",
    )
    parser.add_argument("--max_per_hour", type=int, default=1000, help="Per-username hourly quota")
    parser.add_argument("--max_rows", type=int, default=1, help="GeoNames search maxRows")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--sleep_sec", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--wait_on_quota",
        action="store_true",
        help="Wait for next hour when all usernames are out of quota",
    )
    return parser.parse_args()


def parse_json_obj(text: str) -> Dict:
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        try:
            return json.loads(block.replace("'", '"'))
        except Exception:
            return {}


def extract_location(content: str) -> Tuple[str, str]:
    obj = parse_json_obj(content)
    if obj:
        country = str(obj.get("country", "")).strip()
        city = str(obj.get("city", "")).strip()
        if country or city:
            return country, city

    country_m = re.search(r"country\s*[:=]\s*([^\n,}]+)", content, flags=re.IGNORECASE)
    city_m = re.search(r"city\s*[:=]\s*([^\n,}]+)", content, flags=re.IGNORECASE)
    country = country_m.group(1).strip() if country_m else ""
    city = city_m.group(1).strip() if city_m else ""
    if country or city:
        return country, city

    parts = [x.strip() for x in content.split(",") if x.strip()]
    if len(parts) >= 2:
        return parts[-1], parts[0]
    return "", ""


def _prediction_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if not value:
            return ""
        v = value[0]
        return v.strip() if isinstance(v, str) else str(v).strip()
    if isinstance(value, dict):
        for k in ["raw_response", "predicted_answer", "prediction", "answer", "text"]:
            if k in value:
                return _prediction_to_text(value[k])
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def load_predictions(path: str) -> List[Dict[str, str]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    rows: List[Dict[str, str]] = []

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            name = str(item.get("filename", item.get("image_name", f"idx_{i}")))
            pred = _prediction_to_text(item.get("predicted_answer", item.get("raw_response", "")))
            rows.append({"name": name, "prediction": pred})
        return rows

    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.append({"name": str(k), "prediction": _prediction_to_text(v)})
        return rows

    raise ValueError(f"Unsupported prediction JSON type: {type(obj)}")


class UsernamePool:
    def __init__(self, usernames: List[str], max_per_hour: int):
        now = time.time()
        self.max_per_hour = max_per_hour
        self.stats: Dict[str, Dict[str, float]] = {}
        for u in usernames:
            if u:
                self.stats[u] = {
                    "window_start": now,
                    "used": 0,
                    "invalid": 0,
                    "blocked_until": 0.0,
                }
        if not self.stats:
            raise ValueError("No valid usernames provided")

    def _refresh_hour(self, username: str):
        st = self.stats[username]
        now = time.time()
        if now - st["window_start"] >= 3600:
            st["window_start"] = now
            st["used"] = 0
            st["blocked_until"] = 0.0

    def acquire(self) -> Optional[str]:
        now = time.time()
        for u in list(self.stats.keys()):
            self._refresh_hour(u)
        for u, st in self.stats.items():
            if st["invalid"]:
                continue
            if now < st["blocked_until"]:
                continue
            if st["used"] < self.max_per_hour:
                return u
        return None

    def consume(self, username: str):
        self._refresh_hour(username)
        self.stats[username]["used"] += 1

    def mark_invalid(self, username: str):
        self.stats[username]["invalid"] = 1

    def mark_quota_block(self, username: str):
        st = self.stats[username]
        now = time.time()
        # Block until current hour window resets
        st["blocked_until"] = max(st["blocked_until"], st["window_start"] + 3600)

    def next_reset_wait(self) -> float:
        now = time.time()
        waits = []
        for st in self.stats.values():
            if st["invalid"]:
                continue
            waits.append(max(0.0, st["window_start"] + 3600 - now))
        if not waits:
            return -1.0
        return min(waits)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        now = time.time()
        for u, st in self.stats.items():
            out[u] = {
                "used_in_window": int(st["used"]),
                "invalid": int(st["invalid"]),
                "blocked_seconds_left": max(0.0, st["blocked_until"] - now),
            }
        return out


def geonames_search(query: str, username: str, max_rows: int, timeout: float) -> Dict:
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": query,
        "maxRows": max_rows,
        "username": username,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def classify_status_error(status_obj: Dict) -> str:
    message = str(status_obj.get("message", "")).lower()
    value = int(status_obj.get("value", -1))
    if value == 10 or "user does not exist" in message:
        return "user_invalid"
    if "hourly limit" in message or "credits" in message or value in (18, 19, 20):
        return "quota"
    return "other"


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    usernames = [x.strip() for x in args.usernames.split(",") if x.strip()]
    pool = UsernamePool(usernames, args.max_per_hour)

    os.makedirs(args.output_dir, exist_ok=True)
    geocoded_path = os.path.join(args.output_dir, "predictions_geocoded_geonames.json")
    failed_path = os.path.join(args.output_dir, "predictions_geocoded_geonames_failed.json")
    detail_csv_path = os.path.join(args.output_dir, "predictions_geocoded_geonames.csv")
    username_stats_path = os.path.join(args.output_dir, "geonames_username_stats.json")

    items = load_predictions(args.pred_path)
    print(f"Loaded predictions: {len(items)}")

    geocoded: Dict[str, List[float]] = {}
    failed: Dict[str, str] = {}
    details: List[List] = []

    if os.path.exists(geocoded_path):
        geocoded = json.load(open(geocoded_path, "r", encoding="utf-8"))
        print(f"Resume geocoded entries: {len(geocoded)}")
    if os.path.exists(failed_path):
        failed = json.load(open(failed_path, "r", encoding="utf-8"))

    for idx, item in enumerate(tqdm(items, desc="GeoNames geocoding")):
        name = item["name"]
        pred = (item["prediction"] or "").strip()
        if name in geocoded:
            continue
        if not pred:
            failed[name] = "empty prediction"
            continue

        if args.query_mode == "city_country":
            country, city = extract_location(pred)
            query = f"{city}, {country}".strip(", ").strip()
            if not query:
                query = pred
        else:
            country, city = "", ""
            query = pred

        resolved = False
        while not resolved:
            username = pool.acquire()
            if username is None:
                wait_s = pool.next_reset_wait()
                if wait_s < 0:
                    failed[name] = f"{pred} | all usernames invalid"
                    resolved = True
                    break
                if not args.wait_on_quota:
                    failed[name] = f"{pred} | all usernames out of quota (wait {wait_s:.1f}s)"
                    resolved = True
                    break
                sleep_time = max(1.0, wait_s + 1.0)
                print(f"All usernames out of quota. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            pool.consume(username)
            try:
                data = geonames_search(query, username, args.max_rows, args.timeout)
            except Exception as e:
                failed[name] = f"{pred} | request error: {e}"
                resolved = True
                break

            if "status" in data:
                err_type = classify_status_error(data["status"])
                if err_type == "user_invalid":
                    pool.mark_invalid(username)
                    continue
                if err_type == "quota":
                    pool.mark_quota_block(username)
                    continue
                failed[name] = f"{pred} | api status: {data['status']}"
                resolved = True
                break

            geos = data.get("geonames", [])
            if not geos:
                failed[name] = f"{pred} | empty geonames"
                resolved = True
                break

            g = geos[0]
            try:
                lat = float(g.get("lat"))
                lon = float(g.get("lng"))
            except Exception:
                failed[name] = f"{pred} | invalid lat/lng in response"
                resolved = True
                break

            geocoded[name] = [lat, lon]
            details.append(
                [
                    name,
                    pred,
                    query,
                    city,
                    country,
                    lat,
                    lon,
                    username,
                    g.get("name", ""),
                    g.get("toponymName", ""),
                    g.get("countryName", ""),
                    g.get("countryCode", ""),
                    g.get("adminName1", ""),
                    g.get("fclName", ""),
                    g.get("fcodeName", ""),
                    g.get("population", ""),
                    g.get("geonameId", ""),
                ]
            )
            resolved = True

        if (idx + 1) % args.save_interval == 0:
            save_json(geocoded, geocoded_path)
            save_json(failed, failed_path)
            save_json(pool.summary(), username_stats_path)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    save_json(geocoded, geocoded_path)
    save_json(failed, failed_path)
    save_json(pool.summary(), username_stats_path)

    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "prediction",
                "query",
                "parsed_city",
                "parsed_country",
                "pred_lat",
                "pred_lon",
                "username",
                "name_resp",
                "toponymName",
                "countryName",
                "countryCode",
                "adminName1",
                "fclName",
                "fcodeName",
                "population",
                "geonameId",
            ]
        )
        w.writerows(details)

    print(f"Done. geocoded={len(geocoded)} failed={len(failed)}")
    print(f"Geocoded JSON: {geocoded_path}")
    print(f"Detailed CSV: {detail_csv_path}")
    print(f"Failed JSON: {failed_path}")
    print(f"Username stats: {username_stats_path}")


if __name__ == "__main__":
    main()
