from __future__ import annotations
import json
import math
import os
import re
import subprocess
import tempfile
import time
from urllib.parse import quote_plus
from pathlib import Path
from typing import Any
import requests
from .geoskill_graph_runtime import build_skill_graph_plan
from .skill_library import SkillLibrary
from .skill_parser import COUNTRY_NAME_TO_ISO2, COUNTRY_TO_REGION
from .vlm_client import VLMClient
_JSON_SCHEMA_INSTRUCTION = """You MUST respond with ONLY a valid JSON object in the following format (no markdown, no extra text):
{
  "country": "<full country name>",
  "country_code": "<ISO 3166-1 alpha-2 lowercase code>",
  "region": "<continent or sub-region>",
    "city": "<city or locality, if known>",
    "province_or_state": "<province/state/region, if known>",
    "address": "<final textual location guess for geocoding>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<your step-by-step reasoning>",
  "evidence": ["<visual cue 1>", "<visual cue 2>", ...]
}"""
_ISO2_TO_COUNTRY_NAME = {v: k for k, v in COUNTRY_NAME_TO_ISO2.items()}
_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}
_DEFAULT_GEONAMES_USERNAMES = [
    "yjiang194",
    "arandinglv",
    "ouyangxin",
    "yiqiuliu",
    "ouyang_xin",
    "yiqiuliu2",
    "yiqiuliu3",
]
_GEONAMES_MAX_PER_HOUR = int(os.getenv("GEONAMES_MAX_PER_HOUR", "1000"))
_GEONAMES_USER_STATE: dict[str, dict[str, float]] = {}
_REVERSE_GEOCODE_COUNTRY_CACHE: dict[str, str] = {}
_GEOCODE_TIMEOUT_SECONDS = float(os.getenv("GEOCODE_TIMEOUT_SECONDS", "3.0"))
_REVERSE_GEOCODE_TIMEOUT_SECONDS = float(os.getenv("REVERSE_GEOCODE_TIMEOUT_SECONDS", "2.5"))
_VAGUE_ADDRESS_PATTERNS = [
    r"\bnear\b",
    r"\bnearby\b",
    r"\baround\b",
    r"\boutskirts\b",
    r"\brural\b",
    r"\barea\b",
    r"\bvicinity\b",
    r"\bclose\s+to\b",
    r"\boutside\b",
    r"\bapproximately\b",
]
_GEOLOCATION_SYSTEM = (
    "You are an expert geolocation analyst. You analyze street view images to determine "
    "the precise location. Use visual evidence: road markings, signs, vegetation, architecture, "
    "utility poles, driving side, license plates, terrain, climate indicators. "
    "Always provide your best guess even when uncertain — never refuse to answer."
)

# Extracted own-method predictor core (adapter-free package).

def _multimodal_retrieve_skills(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    base_query: str,
    top_k: int,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    mode = (retrieval_mode or "hybrid").lower()
    if mode not in {"hybrid", "qwen_multimodal", "text_only"}:
        mode = "hybrid"

    if mode == "text_only":
        return skill_library.retrieve(query_text=base_query, top_k=top_k, alpha=alpha, deduplicate_region=False)

    mm_prompt = (
        "Analyze this image and return ONLY JSON with keys: scene_summary, visual_cues, scripts, road, poles, signs, climate. "
        "Keep scene_summary <= 120 words and visual_cues as a short list."
    )
    mm_raw = vlm.query(
        image_path=image_path,
        system_prompt="You are a strict JSON image analyzer.",
        user_prompt=mm_prompt,
        temperature=0.1,
    )
    try:
        mm_obj = VLMClient.extract_json(mm_raw)
        mm_text = json.dumps(mm_obj, ensure_ascii=False)
    except Exception:
        mm_text = mm_raw[:500]

    queries: list[tuple[str, float]] = [(base_query, 0.55), (mm_text, 0.45)]
    if mode == "qwen_multimodal":
        queries = [(base_query, 0.4), (mm_text, 0.6)]

    return skill_library.retrieve_multi(queries=queries, top_k=top_k, alpha=alpha, deduplicate_region=False)

def _parse_json_prediction(raw_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    predicted_address = ""
    try:
        parsed = VLMClient.extract_json(raw_text)
    except (ValueError, json.JSONDecodeError):
        pass

    if parsed and isinstance(parsed, dict):
        country_code = str(parsed.get("country_code", "")).lower().strip()
        country_name = str(parsed.get("country", "")).lower().strip()
        predicted_address = str(
            parsed.get("address", "")
            or parsed.get("final_address", "")
            or parsed.get("location", "")
        ).strip()
        if not country_code or country_code not in COUNTRY_TO_REGION:
            country_code = _country_name_to_iso(country_name)
        lat = _safe_float(parsed.get("lat"))
        lng = _safe_float(parsed.get("lng"))
        if not _is_valid_lat_lng(lat, lng):
            for geocode_query in _build_geocode_queries(parsed=parsed, country_code=country_code):
                geo_latlng = _geocode_text(geocode_query)
                if geo_latlng is not None:
                    lat, lng = geo_latlng
                    break
        conf = _safe_float(parsed.get("confidence"), default=0.5)
        if conf > 1.0:
            conf = conf / 100.0
        reasoning = str(parsed.get("reasoning", ""))
        evidence = parsed.get("evidence", [])
        if isinstance(evidence, list):
            evidence = [str(e) for e in evidence]
        else:
            evidence = []
    else:
        country_code = _extract_country_iso_from_text(raw_text)
        lat = _extract_float_after("lat", raw_text)
        lng = _extract_float_after("lng", raw_text)
        if not _is_valid_lat_lng(lat, lng):
            text_queries = _build_geocode_queries_from_text(raw_text=raw_text, country_code=country_code)
            if text_queries:
                predicted_address = text_queries[0]
            for geocode_query in text_queries:
                latlng = _geocode_text(geocode_query)
                if latlng is not None:
                    lat, lng = latlng
                    break
        conf = _extract_confidence_from_text(raw_text)
        reasoning = raw_text
        evidence = _extract_evidence_spans(raw_text)

    region = COUNTRY_TO_REGION.get(country_code, "unknown")
    return {
        "predicted_country": country_code,
        "predicted_region": region,
        "predicted_address": predicted_address,
        "predicted_lat": lat if lat is not None else math.nan,
        "predicted_lng": lng if lng is not None else math.nan,
        "reasoning_text": raw_text,
        "evidence_spans": evidence,
        "confidence": max(0.0, min(1.0, conf if conf is not None else 0.5)),
    }

def _is_valid_lat_lng(lat: float | None, lng: float | None) -> bool:
    return lat is not None and lng is not None and -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0

def _build_geocode_queries(parsed: dict[str, Any], country_code: str) -> list[str]:
    address = str(parsed.get("address", "")).strip()
    if not address:
        address = str(parsed.get("final_address", "")).strip()
    if not address:
        address = str(parsed.get("location", "")).strip()

    city = str(parsed.get("city", "")).strip()
    state = str(
        parsed.get("province_or_state", "")
        or parsed.get("state", "")
        or parsed.get("province", "")
    ).strip()
    country = str(parsed.get("country", "")).strip()
    if not country and country_code in _ISO2_TO_COUNTRY_NAME:
        country = _ISO2_TO_COUNTRY_NAME[country_code]

    candidates: list[str] = []
    if address and city and country:
        candidates.append(f"{address}, {city}, {country}")
    if address and country:
        candidates.append(f"{address}, {country}")
    if city and state and country:
        candidates.append(f"{city}, {state}, {country}")
    if city and country:
        candidates.append(f"{city}, {country}")
    if state and country:
        candidates.append(f"{state}, {country}")
    if address:
        candidates.append(address)

    deduped: list[str] = []
    seen: set[str] = set()
    for q in candidates:
        qq = " ".join(str(q).replace("\n", " ").split()).strip(" ,")
        if not qq:
            continue
        key = qq.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qq)
    return deduped

def _build_geocode_queries_from_text(raw_text: str, country_code: str) -> list[str]:
    country_hint = _ISO2_TO_COUNTRY_NAME.get(country_code, "")
    parsed: dict[str, Any] | None = None
    try:
        obj = VLMClient.extract_json(raw_text)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        parsed = None
    if parsed:
        return _build_geocode_queries(parsed=parsed, country_code=country_code)

    country_m = re.search(r"country\s*[:=]\s*([^\n,}]+)", raw_text, flags=re.IGNORECASE)
    city_m = re.search(r"city\s*[:=]\s*([^\n,}]+)", raw_text, flags=re.IGNORECASE)
    state_m = re.search(r"(?:state|province|province_or_state)\s*[:=]\s*([^\n,}]+)", raw_text, flags=re.IGNORECASE)
    addr_m = re.search(r"address\s*[:=]\s*([^\n}]+)", raw_text, flags=re.IGNORECASE)

    city = city_m.group(1).strip() if city_m else ""
    state = state_m.group(1).strip() if state_m else ""
    country = country_m.group(1).strip() if country_m else country_hint
    address = addr_m.group(1).strip() if addr_m else ""

    candidates = []
    if address and country:
        candidates.append(f"{address}, {country}")
    if city and state and country:
        candidates.append(f"{city}, {state}, {country}")
    if city and country:
        candidates.append(f"{city}, {country}")
    if country:
        candidates.append(country)

    deduped: list[str] = []
    seen: set[str] = set()
    for q in candidates:
        qq = " ".join(str(q).replace("\n", " ").split()).strip(" ,")
        if not qq:
            continue
        key = qq.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qq)

    if not deduped:
        deduped.append(" ".join(raw_text.split())[:200])
    return deduped

def _geocode_text(query: str | None, timeout_seconds: float = _GEOCODE_TIMEOUT_SECONDS) -> tuple[float, float] | None:
    if not query:
        return None
    q = query.strip()
    if not q:
        return None

    cache_key = q.lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    # 1) Wikipedia title/coordinates API (primary)
    wiki_latlng = _geocode_via_wikipedia(q, timeout_seconds=timeout_seconds)
    if wiki_latlng is not None:
        _GEOCODE_CACHE[cache_key] = wiki_latlng
        return wiki_latlng

    # 2) GeoNames search API (optional; requires GEONAMES_USERNAME)
    geonames_latlng = _geocode_via_geonames(q, timeout_seconds=timeout_seconds)
    if geonames_latlng is not None:
        _GEOCODE_CACHE[cache_key] = geonames_latlng
        return geonames_latlng

    # 3) OSM Nominatim fallback
    try:
        url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={quote_plus(q)}&format=json&limit=1"
        )
        resp = requests.get(
            url,
            headers={"User-Agent": "geoskill-geocoder/1.0"},
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            lat = _safe_float(data[0].get("lat"))
            lng = _safe_float(data[0].get("lon"))
            if _is_valid_lat_lng(lat, lng):
                result = (float(lat), float(lng))
                _GEOCODE_CACHE[cache_key] = result
                return result
    except Exception:
        return None
    return None

def _geocode_via_wikipedia(query: str, timeout_seconds: float = _GEOCODE_TIMEOUT_SECONDS) -> tuple[float, float] | None:
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&format=json&srlimit=1&srsearch={quote_plus(query)}"
        )
        search_resp = requests.get(
            search_url,
            headers={"User-Agent": "geoskill-geocoder/1.0"},
            timeout=timeout_seconds,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        hits = ((search_data.get("query") or {}).get("search") or [])
        if not hits:
            return None

        title = str(hits[0].get("title", "")).strip()
        if not title:
            return None

        coord_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=coordinates&format=json&colimit=1&titles={quote_plus(title)}"
        )
        coord_resp = requests.get(
            coord_url,
            headers={"User-Agent": "geoskill-geocoder/1.0"},
            timeout=timeout_seconds,
        )
        coord_resp.raise_for_status()
        coord_data = coord_resp.json()
        pages = ((coord_data.get("query") or {}).get("pages") or {})
        for page in pages.values():
            coords = page.get("coordinates")
            if not coords:
                continue
            lat = _safe_float(coords[0].get("lat"))
            lng = _safe_float(coords[0].get("lon"))
            if _is_valid_lat_lng(lat, lng):
                return float(lat), float(lng)
    except Exception:
        return None
    return None

def _geocode_via_geonames(query: str, timeout_seconds: float = _GEOCODE_TIMEOUT_SECONDS) -> tuple[float, float] | None:
    usernames = _get_geonames_usernames()
    if not usernames:
        return None

    while True:
        username = _geonames_acquire_username(usernames)
        if not username:
            return None

        _geonames_consume(username)
        try:
            url = (
                "http://api.geonames.org/searchJSON"
                f"?q={quote_plus(query)}&maxRows=1&username={quote_plus(username)}"
            )
            resp = requests.get(
                url,
                headers={"User-Agent": "geoskill-geocoder/1.0"},
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        status = data.get("status") if isinstance(data, dict) else None
        if isinstance(status, dict):
            message = str(status.get("message", "")).lower()
            value = int(status.get("value", -1))
            if value == 10 or "user does not exist" in message:
                _geonames_mark_invalid(username)
                continue
            if "hourly limit" in message or "credits" in message or value in {18, 19, 20}:
                _geonames_mark_quota(username)
                continue
            return None

        rows = data.get("geonames") if isinstance(data, dict) else None
        if isinstance(rows, list) and rows:
            lat = _safe_float(rows[0].get("lat"))
            lng = _safe_float(rows[0].get("lng"))
            if _is_valid_lat_lng(lat, lng):
                return float(lat), float(lng)
        return None
    return None

def _get_geonames_usernames() -> list[str]:
    csv_names = os.getenv("GEONAMES_USERNAMES", "").strip()
    if csv_names:
        vals = [x.strip() for x in csv_names.split(",") if x.strip()]
        if vals:
            return vals

    single = os.getenv("GEONAMES_USERNAME", "").strip()
    if single:
        return [single]

    return [u for u in _DEFAULT_GEONAMES_USERNAMES if u]

def _geonames_refresh(username: str, now: float) -> None:
    state = _GEONAMES_USER_STATE.setdefault(
        username,
        {
            "window_start": now,
            "used": 0.0,
            "invalid": 0.0,
            "blocked_until": 0.0,
        },
    )
    if now - float(state["window_start"]) >= 3600.0:
        state["window_start"] = now
        state["used"] = 0.0
        state["blocked_until"] = 0.0

def _geonames_acquire_username(usernames: list[str]) -> str | None:
    now = time.time()
    for username in usernames:
        _geonames_refresh(username, now)

    for username in usernames:
        state = _GEONAMES_USER_STATE[username]
        if float(state["invalid"]) > 0:
            continue
        if now < float(state["blocked_until"]):
            continue
        if float(state["used"]) < float(_GEONAMES_MAX_PER_HOUR):
            return username
    return None

def _geonames_consume(username: str) -> None:
    now = time.time()
    _geonames_refresh(username, now)
    _GEONAMES_USER_STATE[username]["used"] = float(_GEONAMES_USER_STATE[username]["used"]) + 1.0

def _geonames_mark_invalid(username: str) -> None:
    now = time.time()
    _geonames_refresh(username, now)
    _GEONAMES_USER_STATE[username]["invalid"] = 1.0

def _geonames_mark_quota(username: str) -> None:
    now = time.time()
    _geonames_refresh(username, now)
    state = _GEONAMES_USER_STATE[username]
    state["blocked_until"] = max(float(state["blocked_until"]), float(state["window_start"]) + 3600.0)

def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        f = float(val)
        return f if not math.isnan(f) else default
    except (ValueError, TypeError):
        return default

def _country_name_to_iso(name: str) -> str:
    name = name.lower().strip()
    if name in COUNTRY_NAME_TO_ISO2:
        return COUNTRY_NAME_TO_ISO2[name]
    for full_name, iso in sorted(COUNTRY_NAME_TO_ISO2.items(), key=lambda kv: -len(kv[0])):
        if full_name in name:
            return iso
    return "unknown"

def _extract_country_iso_from_text(text: str) -> str:
    lower = text.lower()
    for pattern in [r"country[:\s]+([a-zA-Z\s]+)", r"\*\*country\*\*[:\s]+([a-zA-Z\s]+)"]:
        m = re.search(pattern, lower)
        if m:
            candidate = m.group(1).strip().split("\n")[0].strip().rstrip(".")
            iso = _country_name_to_iso(candidate)
            if iso != "unknown":
                return iso
    for name, iso in sorted(COUNTRY_NAME_TO_ISO2.items(), key=lambda kv: -len(kv[0])):
        if name in lower:
            return iso
    return "unknown"

def _extract_float_after(label: str, text: str) -> float | None:
    pattern = rf"{label}\s*[:=]\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def _extract_confidence_from_text(text: str) -> float:
    c = _extract_float_after("confidence", text)
    if c is not None:
        return c / 100.0 if c > 1.0 else c
    lower = text.lower()
    if any(w in lower for w in ["100%", "certain", "definitely"]):
        return 0.9
    if any(w in lower for w in ["highly likely", "very likely"]):
        return 0.8
    if any(w in lower for w in ["likely", "probably"]):
        return 0.7
    return 0.6

def _extract_evidence_spans(text: str) -> list[str]:
    spans = []
    for line in text.splitlines():
        line = line.strip(" -•\t*")
        if not line or len(line) < 10:
            continue
        if any(k in line.lower() for k in [
            "road", "sign", "vegetation", "architecture", "pole", "bollard",
            "roof", "line", "tree", "marking", "plate", "flag", "script",
        ]):
            spans.append(line)
    return spans[:12]

def _compact_skills_text(skills: list[dict[str, Any]], max_items: int = 8, max_chars: int = 120) -> str:
    lines: list[str] = []
    for i, skill in enumerate(skills[:max_items]):
        text = str(skill.get("skill_text", "")).replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        region = str(skill.get("region_hint", "unknown"))
        conf = _safe_float(skill.get("confidence"), default=0.0) or 0.0
        lines.append(f"[Skill {i + 1} | {region} | {conf:.2f}] {text}")
    return "\n".join(lines)

def skill_conditioned_v3_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 10,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    stage1_user = (
        "You are an expert GeoGuessr player. Analyze this street view image using the "
        "systematic reasoning strategy that top human players use:\n\n"
        "STEP 1 - HEMISPHERE & CLIMATE ZONE:\n"
        "Look at sun position/shadows to determine hemisphere. Assess climate (tropical, "
        "temperate, arid, continental, polar).\n\n"
        "STEP 2 - CONTINENT NARROWING:\n"
        "Use driving side (left=UK/Aus/Japan/SEAsia, right=Americas/Europe/most of Asia), "
        "road quality, and general landscape to narrow to a continent.\n\n"
        "STEP 3 - COUNTRY IDENTIFICATION:\n"
        "Focus on country-specific markers: utility pole design, bollard style, road "
        "marking patterns, sign design, license plate color/shape, Google car type.\n\n"
        "STEP 4 - REGION WITHIN COUNTRY:\n"
        "Use vegetation patterns, terrain, architectural regional styles, and any visible "
        "text to narrow to a specific region/state/province.\n\n"
        "STEP 5 - FINAL LOCATION TEXT:\n"
        "Based on all evidence, provide your best textual location guess (city/state/address) only.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    stage1_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=stage1_user, temperature=0.2
    )
    stage1_pred = _parse_json_prediction(stage1_out)
    geocot_country = stage1_pred["predicted_country"]
    geocot_region = stage1_pred["predicted_region"]

    local_user = (
        "Focus ONLY on LOCAL DETAILS in this street view image — ignore the big picture:\n"
        "- Utility pole: material (wood/concrete/metal), shape, insulator count/pattern\n"
        "- Road markings: center line color (white/yellow), edge lines, dash patterns\n"
        "- Bollards and guardrails: color, shape, material\n"
        "- Road signs: shape, color scheme, any legible text or script\n"
        "- Vehicles: make/model hints, license plate color and shape\n"
        "- Building construction style and facade material\n"
        "- Any distinctive markings, logos, or labels\n\n"
        'Respond with ONLY JSON: {"pole_type": "...", "road_markings": "...", '
        '"bollards": "...", "signs": "...", "vehicles": "...", "architecture": "...", '
        '"script_type": "...", "implied_country": "...", "implied_region": "..."}'
    )
    local_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=local_user, temperature=0.1
    )
    local_info: dict[str, Any] = {}
    try:
        local_info = VLMClient.extract_json(local_out)
    except (ValueError, json.JSONDecodeError):
        pass

    local_implied_country = str(local_info.get("implied_country", "")).lower().strip()
    local_implied_region = str(local_info.get("implied_region", "")).lower().strip()

    local_iso = _country_name_to_iso(local_implied_country) if local_implied_country else "unknown"
    local_region_from_iso = COUNTRY_TO_REGION.get(local_iso, "unknown")
    if local_implied_region and local_implied_region in {
        "europe", "asia", "north_america", "south_america", "africa", "oceania"
    }:
        local_region = local_implied_region
    elif local_region_from_iso != "unknown":
        local_region = local_region_from_iso
    else:
        local_region = "unknown"

    if local_region != "unknown" and local_region != geocot_region:
        active_region = local_region
        active_country = local_iso if local_iso != "unknown" else geocot_country
    else:
        active_region = geocot_region
        active_country = geocot_country

    synth_query = (
        f"{stage1_out[:500]} "
        f"{json.dumps(local_info)[:300]} "
        f"{active_country} {active_region}"
    )

    all_skills = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=synth_query,
        top_k=top_k * 5,
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )

    region_skills = [s for s in all_skills if s.get("region_hint") == active_region]
    region_skills.sort(key=lambda s: len(s.get("skill_text", "")), reverse=True)
    region_skills = region_skills[:top_k]

    if len(region_skills) < 3:
        region_skills = sorted(all_skills[:top_k], key=lambda s: len(s.get("skill_text", "")), reverse=True)

    skills_text = _compact_skills_text(region_skills, max_items=8, max_chars=100)

    local_str = json.dumps(local_info)[:350] if local_info else local_out[:300]
    cross_val_note = ""
    if local_region != "unknown" and local_region != geocot_region:
        cross_val_note = (
            f"NOTE: Stage 1 suggested {geocot_region}/{geocot_country} but Stage 2 indicates "
            f"{local_region}/{active_country}. Weight local detail heavily.\n"
        )

    stage1_summary = stage1_out[:300]

    final_user = (
        f"Stage 1 reasoning:\n{stage1_summary}\n\n"
        f"Stage 2 local details:\n{local_str}\n"
        f"{cross_val_note}\n"
        f"Expert skills:\n{skills_text}\n\n"
        "Synthesize all evidence. Use local details to pinpoint a final textual location guess (city/state/address).\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    try:
        final_out = vlm.query(
            image_path=None, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=final_user, temperature=0.2
        )
    except Exception:
        fallback_user = (
            f"Candidate country/region: {active_country}/{active_region}\n"
            f"Stage 1 summary: {stage1_summary}\n"
            f"Local details: {local_str[:220]}\n"
            f"Expert skills:\n{skills_text}\n\n"
            + _JSON_SCHEMA_INSTRUCTION
        )
        final_out = vlm.query(
            image_path=image_path,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=fallback_user,
            temperature=0.15,
        )
    pred = _parse_json_prediction(final_out)
    pred["retrieved_skills"] = region_skills
    pred["stage1_prediction"] = stage1_pred
    pred["local_analysis"] = local_info
    pred["active_region"] = active_region
    pred["geocot_region"] = geocot_region
    return pred

def skill_conditioned_v4_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 12,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    # Stabilized fallback: v4 currently issues many sequential VLM calls and is fragile
    # under upstream API jitter. Route through the robust v3 stack to guarantee completion.
    pred = skill_conditioned_v3_predict(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        top_k=max(10, top_k),
        retrieval_mode=retrieval_mode,
    )
    pred["v4_mode"] = "stabilized_via_v3"
    return pred

    # ── Pass A: GeoCoT 5-step (same proven prompt as v2/v3 stage 1) ──────────
    pass_a_user = (
        "You are an expert GeoGuessr player. Analyze this street view image using the "
        "systematic reasoning strategy that top human players use:\n\n"
        "STEP 1 - HEMISPHERE & CLIMATE ZONE: Look at sun position/shadows, assess climate.\n\n"
        "STEP 2 - CONTINENT NARROWING: Use driving side, road quality, landscape.\n\n"
        "STEP 3 - COUNTRY IDENTIFICATION: Focus on utility poles, bollards, road markings, "
        "sign design, license plate color/shape, Google car type.\n\n"
        "STEP 4 - REGION WITHIN COUNTRY: Vegetation, terrain, architecture, visible text.\n\n"
        "STEP 5 - FINAL LOCATION TEXT: Provide your best city/state/address guess.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    pass_a_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_a_user, temperature=0.2
    )
    pred_a = _parse_json_prediction(pass_a_out)

    # ── Pass B: Hierarchical forced (PIGEON-style continent→country→city) ─────
    pass_b_user = (
        "You are an expert GeoGuessr player. Use a STRICT HIERARCHICAL approach:\n\n"
        "LEVEL 1 — CONTINENT (MUST decide first):\n"
        "Look ONLY at: driving side (left-hand=UK/Aus/Japan/SEAsia/India), road surface quality, "
        "vegetation biome (tropical/temperate/boreal/desert/savanna), sky angle/sunlight, "
        "and Google car shadow. Name your continent before anything else.\n\n"
        "LEVEL 2 — COUNTRY (given your continent):\n"
        "Now focus on continent-specific discriminators: utility pole construction (wood/metal/concrete, "
        "insulator count), road marking colors (yellow center = Americas/Korea, white = Europe/Africa/most of Asia), "
        "chevron/bollard colors, sign language/script (Latin/Cyrillic/Arabic/Hangul/Thai/Devanagari), "
        "license plate shape and color, specific road sign designs unique to each country.\n\n"
        "LEVEL 3 — CITY/REGION (given your country):\n"
        "Use climate micro-zone, terrain, architecture style, vegetation density, any legible text "
        "or brand names, urban vs rural density, to narrow to a specific city or province.\n\n"
        "LEVEL 4 — FINAL LOCATION TEXT: Based on levels 1-3, provide city/state/address.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    pass_b_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_b_user, temperature=0.3
    )
    pred_b = _parse_json_prediction(pass_b_out)

    # ── Pass C: Adversarial self-correction ──────────────────────────────────
    # Prime with the most common error pattern: confusing visually similar countries
    pass_c_user = (
        "You are an expert GeoGuessr player tasked with identifying a location from this street view image.\n\n"
        "IMPORTANT: Many locations are deliberately chosen to be ambiguous and look like other countries. "
        "Before committing to an answer, explicitly consider and RULE OUT the most visually similar alternatives:\n\n"
        "1. First, note the single most distinctive feature in the image (the 'smoking gun' clue).\n"
        "2. List the top 3 candidate countries this could be, with the key evidence FOR each.\n"
        "3. For each candidate, identify the SINGLE piece of evidence that would RULE IT OUT.\n"
        "4. Apply this elimination: which candidates are ruled out and why?\n"
        "5. Commit to your final answer with a precise textual location (city/state/address).\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    pass_c_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_c_user, temperature=0.4
    )
    pred_c = _parse_json_prediction(pass_c_out)

    # ── Local detail fingerprint (GRE-style) ─────────────────────────────────
    local_user = (
        "Focus ONLY on LOCAL DETAILS in this street view image — ignore the big picture:\n"
        "- Utility pole: material (wood/concrete/metal), shape, insulator count/pattern\n"
        "- Road markings: center line color (white/yellow), edge lines, dash patterns\n"
        "- Bollards and guardrails: color, shape, material\n"
        "- Road signs: shape, color scheme, any legible text or script\n"
        "- Vehicles: make/model hints, license plate color and shape\n"
        "- Building construction style and facade material\n"
        "- Any distinctive markings, logos, or labels\n\n"
        'Respond with ONLY JSON: {"pole_type": "...", "road_markings": "...", '
        '"bollards": "...", "signs": "...", "vehicles": "...", "architecture": "...", '
        '"script_type": "...", "implied_country": "...", "implied_region": "..."}'
    )
    local_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=local_user, temperature=0.1
    )
    local_info: dict[str, Any] = {}
    try:
        local_info = VLMClient.extract_json(local_out)
    except (ValueError, json.JSONDecodeError):
        pass

    # ── Majority vote on country ──────────────────────────────────────────────
    votes: dict[str, list[int]] = {}  # country_code -> list of pass indices
    for i, pred in enumerate([pred_a, pred_b, pred_c]):
        cc = pred["predicted_country"]
        if cc and cc != "unknown":
            votes.setdefault(cc, []).append(i)

    # Find majority country (2+ votes wins)
    voted_country = "unknown"
    voted_region = "unknown"
    for cc, indices in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        if len(indices) >= 2:
            voted_country = cc
            voted_region = COUNTRY_TO_REGION.get(cc, "unknown")
            break

    # If no majority (all three disagree), use local detail or fall back to pass A
    if voted_country == "unknown":
        local_implied_country = str(local_info.get("implied_country", "")).lower().strip()
        local_iso = _country_name_to_iso(local_implied_country) if local_implied_country else "unknown"
        if local_iso != "unknown":
            voted_country = local_iso
            voted_region = COUNTRY_TO_REGION.get(local_iso, "unknown")
        else:
            voted_country = pred_a["predicted_country"]
            voted_region = pred_a["predicted_region"]

    # ── Skill retrieval on voted region ──────────────────────────────────────
    synth_query = (
        f"{pass_a_out[:400]} {pass_b_out[:200]} "
        f"{json.dumps(local_info)[:250]} "
        f"{voted_country} {voted_region}"
    )
    all_skills = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=synth_query,
        top_k=top_k * 5,
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )
    region_skills = [s for s in all_skills if s.get("region_hint") == voted_region]
    region_skills.sort(key=lambda s: len(s.get("skill_text", "")), reverse=True)
    region_skills = region_skills[:top_k]
    if len(region_skills) < 3:
        region_skills = sorted(all_skills[:top_k], key=lambda s: len(s.get("skill_text", "")), reverse=True)

    skills_text = _compact_skills_text(region_skills, max_items=8, max_chars=100)

    # ── Text-only self-critique + final synthesis ─────────────────────────────
    # Note vote agreement level
    vote_summary = ", ".join([f"Pass {'ABC'[i]}→{pred['predicted_country']}"
                               for i, pred in enumerate([pred_a, pred_b, pred_c])])
    all_agree = len(set(p["predicted_country"] for p in [pred_a, pred_b, pred_c])) == 1
    agreement_note = (
        f"All 3 passes agree: {voted_country}. High confidence." if all_agree
        else f"Vote: {vote_summary}. Majority: {voted_country}."
    )

    # Aggregate geocoded points from agreeing passes (model itself does not output coordinates)
    agreeing_preds = [p for p in [pred_a, pred_b, pred_c]
                      if p["predicted_country"] == voted_country
                      and not math.isnan(p.get("predicted_lat", math.nan))
                      and not math.isnan(p.get("predicted_lng", math.nan))]
    avg_lat = sum(p["predicted_lat"] for p in agreeing_preds) / len(agreeing_preds) if agreeing_preds else math.nan
    avg_lng = sum(p["predicted_lng"] for p in agreeing_preds) / len(agreeing_preds) if agreeing_preds else math.nan

    # Summarize pass reasoning (truncated)
    pass_summaries = (
        f"Pass A (GeoCoT): country={pred_a['predicted_country']}, "
        f"reasoning={pass_a_out[:250]}\n\n"
        f"Pass B (Hierarchical): country={pred_b['predicted_country']}, "
        f"reasoning={pass_b_out[:200]}\n\n"
        f"Pass C (Adversarial): country={pred_c['predicted_country']}, "
        f"key elimination={pass_c_out[:200]}"
    )
    local_str = json.dumps(local_info)[:300] if local_info else local_out[:250]

    critique_user = (
        f"You ran 3 independent geolocation analyses of the same street view image. "
        f"{agreement_note}\n\n"
        f"{pass_summaries}\n\n"
        f"Local detail fingerprint: {local_str}\n\n"
        f"Expert geographic skills for {voted_region}:\n{skills_text}\n\n"
        "TASK: Critically evaluate the 3 passes. "
        "1. Which pass made the strongest case and why? "
        "2. Are there any contradictions with the local detail fingerprint? "
        "3. Do the expert skills confirm or challenge the voted country? "
        "4. What is your final refined textual location answer (city/state/address)?\n\n"
        "Output textual location fields only.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    try:
        final_out = vlm.query(
            image_path=None,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=critique_user,
            temperature=0.1,
        )
    except Exception:
        fallback_user = (
            f"Vote summary: {vote_summary}. Voted country/region: {voted_country}/{voted_region}.\n"
            f"Local detail fingerprint: {local_str[:220]}\n"
            f"Expert skills:\n{skills_text}\n\n"
            + _JSON_SCHEMA_INSTRUCTION
        )
        final_out = vlm.query(
            image_path=image_path,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=fallback_user,
            temperature=0.1,
        )
    pred_final = _parse_json_prediction(final_out)
    pred_final["pass_a"] = pred_a
    pred_final["pass_b"] = pred_b
    pred_final["pass_c"] = pred_c
    pred_final["local_analysis"] = local_info
    pred_final["voted_country"] = voted_country
    pred_final["voted_region"] = voted_region
    pred_final["vote_summary"] = vote_summary
    pred_final["retrieved_skills"] = region_skills
    pred_final["avg_lat_from_votes"] = avg_lat
    pred_final["avg_lng_from_votes"] = avg_lng
    return pred_final
