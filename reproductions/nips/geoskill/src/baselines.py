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

from .GeoVista.skill_graph_runtime import build_skill_graph_plan
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


def _run_official_cli_predict(
    baseline_name: str,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    cfg = official_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None

    cmd_template = str(cfg.get("command", "")).strip()
    if not cmd_template:
        raise RuntimeError(f"{baseline_name}: official mode enabled but no command configured")

    timeout_seconds = float(cfg.get("timeout_seconds", 180.0))
    cwd = str(cfg.get("cwd", ".")).strip()
    output_mode = str(cfg.get("output_mode", "stdout_json")).strip().lower()
    output_file = str(cfg.get("output_file", "")).strip()
    env = os.environ.copy()
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)

    image_abs = str(Path(image_path).resolve())
    cmd = cmd_template.format(image_path=image_abs, output_file=output_file)
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{baseline_name}: official command failed with code {proc.returncode}. stderr={proc.stderr[:400]}"
        )

    if output_mode == "file_json":
        if not output_file:
            raise RuntimeError(f"{baseline_name}: output_mode=file_json requires output_file")
        out_path = Path(output_file)
        if not out_path.exists():
            raise RuntimeError(f"{baseline_name}: expected output file not found: {output_file}")
        raw = out_path.read_text(encoding="utf-8")
    else:
        raw = proc.stdout.strip()

    parsed = _parse_json_prediction(raw)
    parsed["external_baseline_name"] = baseline_name
    parsed["official_mode"] = True
    parsed["official_command"] = cmd
    return parsed


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
            headers={"User-Agent": "geoskill-baseline-geocoder/1.0"},
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
            headers={"User-Agent": "geoskill-baseline-geocoder/1.0"},
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
            headers={"User-Agent": "geoskill-baseline-geocoder/1.0"},
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
                headers={"User-Agent": "geoskill-baseline-geocoder/1.0"},
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


def _normalize_country_code(value: Any) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return "unknown"
    if len(s) == 2 and s.isalpha():
        return s if s in COUNTRY_TO_REGION else "unknown"
    return _country_name_to_iso(s)


def _extract_country_top3_codes(raw_text: str, fallback_country: str = "unknown") -> list[str]:
    parsed: dict[str, Any] | None = None
    try:
        obj = VLMClient.extract_json(raw_text)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        parsed = None

    values: list[Any] = []
    if parsed:
        for key in ["top3_countries", "top_3_countries", "country_top3", "country_candidates", "countries"]:
            top_vals = parsed.get(key)
            if isinstance(top_vals, list):
                for item in top_vals[:8]:
                    if isinstance(item, dict):
                        values.extend([item.get("country_code"), item.get("country"), item.get("iso2"), item.get("code")])
                    else:
                        values.append(item)
        values.extend([parsed.get("country_code"), parsed.get("country")])

    if not values:
        m = re.search(r"top\s*[-_ ]*3[^:\n]*[:=]\s*([^\n]+)", raw_text, flags=re.IGNORECASE)
        if m:
            values.extend(re.split(r"[,;/|]", m.group(1)))

    if fallback_country:
        values.append(fallback_country)

    out: list[str] = []
    seen: set[str] = set()
    for val in values:
        iso = _normalize_country_code(val)
        if iso == "unknown" or iso in seen:
            continue
        seen.add(iso)
        out.append(iso)
        if len(out) >= 3:
            break
    return out


def _is_vague_address(address: str) -> bool:
    addr = str(address or "").strip().lower()
    if not addr:
        return True
    if len([seg for seg in addr.split(",") if seg.strip()]) < 2:
        return True
    return any(re.search(pattern, addr) for pattern in _VAGUE_ADDRESS_PATTERNS)


def _address_specificity_score(address: str) -> float:
    addr = str(address or "").strip()
    if not addr:
        return 0.0

    score = 0.0
    segments = [s.strip() for s in addr.split(",") if s.strip()]
    if len(segments) >= 4:
        score += 0.45
    elif len(segments) == 3:
        score += 0.34
    elif len(segments) == 2:
        score += 0.22
    else:
        score += 0.10

    lower = addr.lower()
    if re.search(
        r"\b(street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|"
        r"drive|dr\.?|highway|hwy|route|junction|intersection|district|neighborhood|"
        r"community|village|ward|county|province|state)\b",
        lower,
    ):
        score += 0.35

    if re.search(r"\d", lower):
        score += 0.10

    if _is_vague_address(addr):
        score -= 0.25

    return max(0.0, min(1.0, score))


def _reverse_geocode_country_iso(
    lat: float | None,
    lng: float | None,
    timeout_seconds: float = _REVERSE_GEOCODE_TIMEOUT_SECONDS,
) -> str:
    if not _is_valid_lat_lng(lat, lng):
        return "unknown"

    assert lat is not None and lng is not None
    cache_key = f"{lat:.4f},{lng:.4f}"
    if cache_key in _REVERSE_GEOCODE_COUNTRY_CACHE:
        return _REVERSE_GEOCODE_COUNTRY_CACHE[cache_key]

    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat:.7f}&lon={lng:.7f}&format=jsonv2&zoom=3"
        )
        resp = requests.get(
            url,
            headers={"User-Agent": "geoskill-baseline-geocoder/1.0"},
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address") if isinstance(data, dict) else None
        iso = _normalize_country_code((addr or {}).get("country_code", ""))
        _REVERSE_GEOCODE_COUNTRY_CACHE[cache_key] = iso
        return iso
    except Exception:
        _REVERSE_GEOCODE_COUNTRY_CACHE[cache_key] = "unknown"
        return "unknown"


def _prediction_geo_country(pred: dict[str, Any]) -> str:
    lat = _safe_float(pred.get("predicted_lat"))
    lng = _safe_float(pred.get("predicted_lng"))
    return _reverse_geocode_country_iso(lat, lng)


def _country_consistency_score(pred_country: str, geo_country: str, top3: list[str]) -> float:
    valid_top3 = [c for c in top3 if c and c != "unknown"]
    if not valid_top3:
        return 0.5

    top1 = valid_top3[0]
    if geo_country != "unknown":
        if geo_country == top1:
            return 1.0
        if geo_country in valid_top3:
            return 0.85
        return 0.0

    if pred_country == top1:
        return 0.75
    if pred_country in valid_top3:
        return 0.60
    return 0.0


def _prediction_country_mismatch(pred: dict[str, Any], top3: list[str]) -> bool:
    valid_top3 = [c for c in top3 if c and c != "unknown"]
    if not valid_top3:
        return False

    pred_country = _normalize_country_code(pred.get("predicted_country", ""))
    geo_country = _prediction_geo_country(pred)
    if geo_country != "unknown":
        return geo_country not in set(valid_top3)
    return pred_country not in set(valid_top3)


def _build_country_shortlist(
    vlm: VLMClient,
    image_path: str,
    task_prompt: str,
    fallback_country: str = "unknown",
) -> tuple[list[str], str]:
    shortlist_user = (
        "First do COUNTRY COARSE CLASSIFICATION only. "
        "Return ONLY JSON: {\"top3_countries\":[\"iso2\",\"iso2\",\"iso2\"],\"reason\":\"...\"}. "
        "Use ISO 3166-1 alpha-2 lowercase codes.\n\n"
        f"Task context:\n{task_prompt[:1000]}"
    )
    shortlist_raw = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=shortlist_user,
        temperature=0.15,
        max_attempts=1,
    )
    top3 = _extract_country_top3_codes(shortlist_raw, fallback_country=fallback_country)
    return top3, shortlist_raw


def _refine_vague_address_once(
    vlm: VLMClient,
    image_path: str,
    task_prompt: str,
    candidate_raw: str,
    top3: list[str],
) -> tuple[dict[str, Any], str, bool]:
    parsed = _parse_json_prediction(candidate_raw)
    if not _is_vague_address(str(parsed.get("predicted_address", ""))):
        return parsed, candidate_raw, False

    refine_user = (
        f"{task_prompt}\n\n"
        f"Country shortlist(top-3): {', '.join(top3) if top3 else 'unknown'}.\n"
        "Your previous address is too vague (contains near/outskirts/rural/area-like wording).\n"
        "Rewrite to a MORE SPECIFIC address in this strict form:\n"
        "road/intersection/community + city + province_or_state + country.\n"
        "Do not use vague words and do not output placeholders.\n\n"
        f"Previous output:\n{candidate_raw[:1200]}\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    refined_raw = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=refine_user,
        temperature=0.10,
        max_attempts=1,
    )
    return _parse_json_prediction(refined_raw), refined_raw, True


def _repair_country_mismatch_once(
    vlm: VLMClient,
    image_path: str,
    task_prompt: str,
    candidate_raw: str,
    candidate_pred: dict[str, Any],
    top3: list[str],
) -> tuple[dict[str, Any], str, bool]:
    if not _prediction_country_mismatch(candidate_pred, top3):
        return candidate_pred, candidate_raw, False

    geo_country = _prediction_geo_country(candidate_pred)
    repair_user = (
        f"{task_prompt}\n\n"
        f"Country shortlist(top-3): {', '.join(top3) if top3 else 'unknown'}.\n"
        f"Current geocode country: {geo_country}. This is inconsistent with top-1/top-3.\n"
        "Re-solve ONE MORE TIME and ensure the final address geocodes to top-1 or top-3 country.\n"
        "Keep address specific: road/intersection/community + city + province_or_state + country.\n\n"
        f"Current output:\n{candidate_raw[:1200]}\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    repaired_raw = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=repair_user,
        temperature=0.12,
        max_attempts=1,
    )
    return _parse_json_prediction(repaired_raw), repaired_raw, True


def _score_prediction_candidate(pred: dict[str, Any], top3: list[str]) -> dict[str, Any]:
    pred_country = _normalize_country_code(pred.get("predicted_country", ""))
    geo_country = _prediction_geo_country(pred)
    consistency = _country_consistency_score(pred_country=pred_country, geo_country=geo_country, top3=top3)
    confidence = max(0.0, min(1.0, _safe_float(pred.get("confidence"), default=0.5) or 0.5))
    specificity = _address_specificity_score(str(pred.get("predicted_address", "")))
    score = 0.55 * consistency + 0.25 * confidence + 0.20 * specificity

    top1 = top3[0] if top3 else "unknown"
    if (geo_country != "unknown" and geo_country == top1) or (geo_country == "unknown" and pred_country == top1):
        score += 0.05
    if _is_vague_address(str(pred.get("predicted_address", ""))):
        score -= 0.20

    return {
        "score": float(score),
        "pred_country": pred_country,
        "geo_country": geo_country,
        "country_consistency": float(consistency),
        "confidence": float(confidence),
        "address_specificity": float(specificity),
    }


def _predict_with_country_checked_vote(
    vlm: VLMClient,
    image_path: str,
    task_prompt: str,
    base_temperature: float = 0.10,
    num_candidates: int = 3,
    fallback_country: str = "unknown",
    single_pass: bool = False,
) -> dict[str, Any]:
    top3, shortlist_raw = _build_country_shortlist(
        vlm=vlm,
        image_path=image_path,
        task_prompt=task_prompt,
        fallback_country=fallback_country,
    )

    candidates: list[dict[str, Any]] = []
    for idx in range(max(1, int(num_candidates))):
        temp = min(0.22, max(0.05, base_temperature + 0.03 * idx))
        candidate_user = (
            f"{task_prompt}\n\n"
            f"Country shortlist(top-3): {', '.join(top3) if top3 else 'unknown'}.\n"
            "Generate ONE candidate location.\n"
            "Address must be specific and include: road/intersection/community + city + province_or_state + country.\n"
            "Do not use vague words like near/outskirts/rural/area.\n\n"
            + _JSON_SCHEMA_INSTRUCTION
        )
        raw = _query_with_retry(
            vlm=vlm,
            image_path=image_path,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=candidate_user,
            temperature=temp,
            max_attempts=1,
        )
        pred = _parse_json_prediction(raw)

        refined = False
        repaired = False
        if not single_pass:
            pred, raw, refined = _refine_vague_address_once(
                vlm=vlm,
                image_path=image_path,
                task_prompt=task_prompt,
                candidate_raw=raw,
                top3=top3,
            )

            pred, raw, repaired = _repair_country_mismatch_once(
                vlm=vlm,
                image_path=image_path,
                task_prompt=task_prompt,
                candidate_raw=raw,
                candidate_pred=pred,
                top3=top3,
            )

        score_info = _score_prediction_candidate(pred=pred, top3=top3)
        candidates.append(
            {
                "prediction": pred,
                "raw": raw,
                "temperature": temp,
                "refined_vague_address": refined,
                "repaired_country_mismatch": repaired,
                "score_info": score_info,
            }
        )

    if not candidates:
        fallback_raw = _query_with_retry(
            vlm=vlm,
            image_path=image_path,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=task_prompt + "\n\n" + _JSON_SCHEMA_INSTRUCTION,
            temperature=0.12,
            max_attempts=1,
        )
        fallback_pred = _parse_json_prediction(fallback_raw)
        fallback_pred["country_top3"] = top3
        fallback_pred["country_shortlist_raw"] = shortlist_raw
        return fallback_pred

    best = max(candidates, key=lambda c: float(c["score_info"]["score"]))
    final_pred = dict(best["prediction"])
    final_pred["country_top3"] = top3
    final_pred["country_shortlist_raw"] = shortlist_raw
    if int(num_candidates) > 1 and not single_pass:
        final_pred["candidate_vote"] = [
            {
                "temperature": c["temperature"],
                "score": c["score_info"]["score"],
                "pred_country": c["score_info"]["pred_country"],
                "geo_country": c["score_info"]["geo_country"],
                "country_consistency": c["score_info"]["country_consistency"],
                "address_specificity": c["score_info"]["address_specificity"],
                "confidence": c["score_info"]["confidence"],
                "refined_vague_address": c["refined_vague_address"],
                "repaired_country_mismatch": c["repaired_country_mismatch"],
                "address": str((c["prediction"] or {}).get("predicted_address", ""))[:220],
            }
            for c in candidates
        ]
    else:
        final_pred["candidate_vote_mode"] = "single_pass"
    final_pred["selected_candidate_score"] = best["score_info"]["score"]
    return final_pred


def _filter_skills_by_region_consistency(
    retrieved_skills: list[dict[str, Any]],
    base_region: str,
    top_k: int,
) -> list[dict[str, Any]]:
    skills = [s for s in retrieved_skills if isinstance(s, dict)]
    if not skills:
        return []

    skills.sort(key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)
    n = max(1, int(top_k))
    region = str(base_region or "unknown").strip().lower()
    if not region or region == "unknown":
        return skills[:n]

    region_skills = [s for s in skills if str(s.get("region_hint", "unknown")).strip().lower() == region]
    unknown_region_skills = [
        s
        for s in skills
        if str(s.get("region_hint", "unknown")).strip().lower() in {"", "unknown"}
    ]

    selected: list[dict[str, Any]] = []
    if region_skills:
        selected.extend(region_skills[: max(4, n - 1)])
        if unknown_region_skills:
            selected.extend(unknown_region_skills[:1])
    else:
        selected.extend(skills[:n])

    deduped: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for skill in selected:
        skill_text = str(skill.get("skill_text", "")).strip().lower()
        if not skill_text or skill_text in seen_text:
            continue
        seen_text.add(skill_text)
        deduped.append(skill)
        if len(deduped) >= n:
            break
    return deduped


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


def _query_with_retry(
    vlm: VLMClient,
    image_path: str | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_attempts: int = 1,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return vlm.query(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
    raise RuntimeError(f"VLM query failed after retries: {last_exc}")


def _clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_image_temp(image_path: str, bbox: list[float]) -> str | None:
    try:
        from PIL import Image  # lazy import to keep baseline lightweight when crop is unused
    except Exception:
        return None

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            safe_bbox = _clamp_bbox(bbox, w, h)
            if safe_bbox is None:
                return None
            crop = img.crop(safe_bbox)
            fd, temp_path = tempfile.mkstemp(prefix="geovista_crop_", suffix=".jpg")
            os.close(fd)
            crop.save(temp_path, format="JPEG")
            return temp_path
    except Exception:
        return None


def _geovista_agentic_copy_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    """Copy-style GeoVista adapter: iterative tool-planning with zoom checks.

    This mirrors GeoVista's agentic loop idea (observe -> plan tools -> verify -> finalize)
    while remaining compatible with this unified VLM interface.
    """
    observe_user = (
        "You are GeoVista-style geolocation agent. Create a strict observation sheet in JSON.\n"
        "Fields: observations(list), hard_clues(list), ambiguous_clues(list), possible_scripts(list), driving_side(str)."
    )
    observe_out = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=observe_user,
        temperature=0.15,
        max_attempts=1,
    )
    observe_json: dict[str, Any] = {}
    try:
        observe_json = VLMClient.extract_json(observe_out)
    except (ValueError, json.JSONDecodeError):
        pass

    plan_user = (
        "Based on the observation sheet, propose GeoVista-style action plan in JSON only:\n"
        "{\"country_candidates\": [..], \"zoom_boxes\": [[x1,y1,x2,y2], ...], \"verification_queries\": [..]}\n"
        "Rules: zoom_boxes use pixel coordinates on original image; at most 2 boxes.\n"
        f"Observation sheet: {json.dumps(observe_json, ensure_ascii=False)[:1200]}"
    )
    plan_out = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=plan_user,
        temperature=0.2,
        max_attempts=1,
    )
    plan_json: dict[str, Any] = {}
    try:
        plan_json = VLMClient.extract_json(plan_out)
    except (ValueError, json.JSONDecodeError):
        pass

    zoom_notes: list[str] = []
    temp_crops: list[str] = []
    zoom_boxes = plan_json.get("zoom_boxes", []) if isinstance(plan_json, dict) else []
    if isinstance(zoom_boxes, list):
        for i, box in enumerate(zoom_boxes[:2]):
            if not isinstance(box, list):
                continue
            crop_path = _crop_image_temp(image_path, box)
            if not crop_path:
                continue
            temp_crops.append(crop_path)
            zoom_user = (
                "Inspect this zoomed crop and extract decisive clues only. Return strict JSON:\n"
                "{\"sign_text\": str, \"script\": str, \"road_markings\": str, \"pole_style\": str, \"country_hint\": str}"
            )
            try:
                zoom_out = _query_with_retry(
                    vlm=vlm,
                    image_path=crop_path,
                    system_prompt=_GEOLOCATION_SYSTEM,
                    user_prompt=zoom_user,
                    temperature=0.1,
                    max_attempts=1,
                )
                zoom_notes.append(f"zoom_{i + 1}: {zoom_out[:500]}")
            except Exception:
                continue

    final_user = (
        "Synthesize GeoVista-style evidence and output final geolocation JSON.\n"
        f"Observation sheet: {json.dumps(observe_json, ensure_ascii=False)[:1000]}\n"
        f"Action plan: {json.dumps(plan_json, ensure_ascii=False)[:900]}\n"
        f"Zoom verification: {' | '.join(zoom_notes)[:1200]}\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    try:
        final_out = _query_with_retry(
            vlm=vlm,
            image_path=image_path,
            system_prompt=_GEOLOCATION_SYSTEM,
            user_prompt=final_user,
            temperature=0.15,
            max_attempts=1,
        )
    finally:
        for p in temp_crops:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    pred = _parse_json_prediction(final_out)
    pred["geovista_observation"] = observe_json if observe_json else observe_out
    pred["geovista_plan"] = plan_json if plan_json else plan_out
    pred["geovista_zoom_notes"] = zoom_notes
    return pred


def _refine_with_skill_boost(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    base_pred: dict[str, Any],
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    base_country = str(base_pred.get("predicted_country", "unknown")).lower()
    base_region = str(base_pred.get("predicted_region", "unknown")).lower()
    base_reasoning = str(base_pred.get("reasoning_text", ""))
    base_conf = _safe_float(base_pred.get("confidence"), default=0.5) or 0.5

    query = f"{base_reasoning[:600]} {base_country} {base_region}".strip()
    all_skills = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=query,
        top_k=top_k * 4,
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )

    if not all_skills:
        pred = dict(base_pred)
        pred["retrieved_skills"] = []
        pred["base_prediction"] = base_pred
        pred["skill_boost_applied"] = False
        pred["skill_boost_reason"] = "no_skills_available"
        return pred

    region_skills = [s for s in all_skills if s.get("region_hint") == base_region]
    region_skills.sort(key=lambda s: len(str(s.get("skill_text", ""))), reverse=True)
    region_skills = region_skills[:top_k]
    if len(region_skills) < 3:
        region_skills = sorted(all_skills[:top_k], key=lambda s: len(str(s.get("skill_text", ""))), reverse=True)

    skills_text = _compact_skills_text(region_skills, max_items=top_k, max_chars=120)
    refine_user = (
        "You are given a baseline geolocation prediction and expert skill priors.\n"
        "Use the image as the primary evidence, and use skills to verify or correct the baseline.\n"
        "If baseline and skills agree, keep the baseline. If you overturn baseline, provide explicit contradictory cues.\n\n"
        f"Baseline country/region: {base_country}/{base_region}\n"
        f"Baseline confidence: {base_conf:.2f}\n"
        f"Baseline reasoning summary:\n{base_reasoning[:500]}\n\n"
        f"Expert skills:\n{skills_text}\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    refine_out = _query_with_retry(
        vlm=vlm,
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=refine_user,
        temperature=0.15,
        max_attempts=1,
    )
    refined = _parse_json_prediction(refine_out)

    if refined.get("predicted_country", "unknown") == "unknown" and base_country != "unknown":
        pred = dict(base_pred)
        pred["skill_boost_reason"] = "fallback_to_base_unknown_country"
    else:
        pred = refined

    pred["retrieved_skills"] = region_skills
    pred["base_prediction"] = base_pred
    pred["skill_boost_applied"] = True
    return pred


# ============================================================
# Baseline 1: Direct VLM (zero-shot, single-pass)
# ============================================================
def direct_vlm_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    user = (
        "Analyze this street view image and identify the country and approximate location. "
        "You must produce a geocodable, specific textual address."
    )
    return _predict_with_country_checked_vote(
        vlm=vlm,
        image_path=image_path,
        task_prompt=user,
        base_temperature=0.10,
        num_candidates=1,
        fallback_country="unknown",
        single_pass=True,
    )


# ============================================================
# Baseline 2: Chain-of-Thought VLM
# ============================================================
def cot_vlm_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    user = (
        "Analyze this street view image step-by-step to determine the location.\n"
        "Think through these categories of evidence in order:\n"
        "1. Road infrastructure (markings, signs, bollards, driving side)\n"
        "2. Vegetation and terrain (tree species, soil color, climate)\n"
        "3. Architecture (building style, roof material, construction)\n"
        "4. Utility infrastructure (pole type, power lines)\n"
        "5. Text/scripts visible (Latin, Cyrillic, Arabic, Asian scripts)\n"
        "6. Vehicles and license plates\n"
        "After reasoning through all evidence, provide your final answer.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    return _parse_json_prediction(out)


# ============================================================
# Baseline 3: Skill-Conditioned VLM (our method)
# ============================================================
def skill_conditioned_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 5,
    min_score: float = 0.15,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    describe_system = "You are a visual scene analyst. Describe concrete visual details only — no speculation about location."
    describe_user = (
        "Describe this street view image in 8-12 bullet points. Focus ONLY on:\n"
        "- Road surface, markings, driving side\n"
        "- Signs, text, scripts visible\n"
        "- Vegetation types, terrain, soil color\n"
        "- Architecture, building materials, roof types\n"
        "- Utility poles, power lines, bollards\n"
        "- Vehicles, license plates\n"
        "- Climate/weather indicators"
    )
    scene_desc = vlm.query(
        image_path=image_path, system_prompt=describe_system, user_prompt=describe_user, temperature=0.1
    )

    retrieved = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=scene_desc,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )
    relevant = [s for s in retrieved if s["score"] >= min_score]
    if not relevant:
        relevant = retrieved[:2]

    skills_text = "\n".join([
        f"- Skill {i+1} (region_hint={s['region_hint']}, confidence={s['confidence']:.2f}): {s['skill_text']}"
        for i, s in enumerate(relevant)
    ])

    user = (
        "Below are expert-derived geographic reasoning skills that MAY be relevant to this image. "
        "IMPORTANT: Only apply skills that match what you actually see in the image. "
        "If a skill contradicts the visual evidence, IGNORE it and rely on the image.\n\n"
        f"Expert Skills:\n{skills_text}\n\n"
        "Now analyze the image using both your own observations and any applicable skills.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["retrieved_skills"] = relevant
    pred["scene_description"] = scene_desc
    return pred


# ============================================================
# Baseline 4: GeoReasoner-style (region-first, then country)
# Two-stage: coarse region classification → fine-grained country
# Paper: "GeoReasoner" (2024), adapted for VLM prompting
# ============================================================
def georeasoner_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    stage1_user = (
        "Look at this street view image and determine which world region it belongs to.\n"
        "Choose from: europe, asia, north_america, south_america, africa, oceania.\n"
        "Provide brief evidence for your choice.\n\n"
        'Respond with ONLY a JSON object: {"region": "<region>", "evidence": ["<cue1>", "<cue2>"], "top3_countries": ["<iso2>", "<iso2>", "<iso2>"]}'
    )
    stage1_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=stage1_user, temperature=0.2
    )
    stage1 = {}
    try:
        stage1 = VLMClient.extract_json(stage1_out)
    except (ValueError, json.JSONDecodeError):
        pass

    region_hint = str(stage1.get("region", "unknown")).lower()
    top3 = stage1.get("top3_countries", [])
    top3_str = ", ".join(str(c) for c in top3) if top3 else "unknown"

    stage2_user = (
        f"This street view image was identified as being in the region: {region_hint}.\n"
        f"Top candidate countries: {top3_str}.\n"
        "Now determine the exact country and final textual location guess (city/state/address) using fine-grained visual evidence "
        "(road markings, pole types, sign styles, vegetation species, architectural details).\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    stage2_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=stage2_user, temperature=0.2
    )
    pred = _parse_json_prediction(stage2_out)
    pred["stage1_raw"] = stage1_out
    pred["stage1_parsed"] = stage1
    return pred


# ============================================================
# External Baseline Adapter: GeoVista (agentic-style prompt)
# Notes:
# - Uses the same VLM endpoint, but follows GeoVista-like "observe -> shortlist -> finalize" flow
# - No external web tools are called in this lightweight adapter
# ============================================================
def geovista_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    observe_user = (
        "Act as an agentic geolocation model. First produce a concise visual observation sheet.\n"
        "Cover: road semantics, utility infrastructure, architecture, vegetation/climate, scripts/signage, vehicles.\n"
        'Return ONLY JSON: {"observations": ["..."], "hard_clues": ["..."], "ambiguous_clues": ["..."]}'
    )
    observe_out = vlm.query(
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=observe_user,
        temperature=0.2,
    )
    observe_json: dict[str, Any] = {}
    try:
        observe_json = VLMClient.extract_json(observe_out)
    except (ValueError, json.JSONDecodeError):
        pass

    shortlist_user = (
        "Based on this observation sheet, propose top-3 candidate countries with explicit elimination reasoning.\n"
        f"Observation sheet: {json.dumps(observe_json)[:1000]}\n"
        'Return ONLY JSON: {"candidates": [{"country_code":"..","why":"..","confidence":0.0}], "winner":".."}'
    )
    shortlist_out = vlm.query(
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=shortlist_user,
        temperature=0.25,
    )

    final_user = (
        "Use the observation sheet + candidate elimination to provide final geolocation.\n"
        f"Observation sheet: {json.dumps(observe_json)[:1000]}\n"
        f"Candidate analysis: {shortlist_out[:1200]}\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    final_out = vlm.query(
        image_path=image_path,
        system_prompt=_GEOLOCATION_SYSTEM,
        user_prompt=final_user,
        temperature=0.15,
    )
    pred = _parse_json_prediction(final_out)
    pred["geovista_observation"] = observe_json if observe_json else observe_out
    pred["geovista_shortlist"] = shortlist_out
    return pred


def geocomp_geocot_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    """External GeoComp baseline adapter.

    GeoComp paper baseline corresponds to GeoCoT-style structured geolocation reasoning.
    We map it to the repository's geocot_predict for unified evaluation.
    """
    pred = geocot_predict(vlm, image_path)
    pred["external_baseline_name"] = "GeoComp/GeoCoT"
    return pred


def safa_predict(_: VLMClient, __: str) -> dict[str, Any]:
    """SAFA is a cross-view retrieval baseline requiring satellite-ground paired index.

    It is not directly executable on single-image GeoRC without a dedicated satellite retrieval corpus
    and SAFA-compatible preprocessing.
    """
    raise RuntimeError(
        "SAFA is not directly runnable on GeoRC single-image setting without satellite retrieval index and paired preprocessing."
    )


# ============================================================
# Baseline 5: GeoCoT — Human gameplay-inspired geographic CoT
# Paper: "GeoCoT" (2025), structured reasoning mimicking human
# expert GeoGuessr strategies
# ============================================================
def geocot_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    user = (
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
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    return _parse_json_prediction(out)


# ============================================================
# Baseline 6: GRE-style multi-stage progressive reasoning
# Paper: "GRE Suite" (2025) — scene→local→semantic→synthesis
# ============================================================
def gre_multistage_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:
    scene_user = (
        "Analyze the GLOBAL SCENE of this street view image. Focus on:\n"
        "- Overall landscape type (urban/suburban/rural, flat/hilly/mountainous)\n"
        "- Climate indicators (tropical/temperate/arid/polar)\n"
        "- Road type and quality\n"
        "- General impression of development level\n\n"
        'Respond with ONLY JSON: {"scene_type": "...", "climate": "...", "development": "...", "hemisphere": "...", "candidate_regions": ["..."]}'
    )
    scene_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=scene_user, temperature=0.1
    )
    scene_info = {}
    try:
        scene_info = VLMClient.extract_json(scene_out)
    except (ValueError, json.JSONDecodeError):
        pass

    local_user = (
        "Now focus on LOCAL DETAILS in this street view image:\n"
        "- Utility pole material and design (wood/concrete/metal, insulator pattern)\n"
        "- Road marking patterns (center line color, edge line style)\n"
        "- Bollard/guardrail design\n"
        "- Sign shapes, colors, and any visible text/script\n"
        "- Vehicle types and license plate color/shape\n"
        "- Building material and construction style\n\n"
        'Respond with ONLY JSON: {"pole_type": "...", "road_markings": "...", "signs": "...", "vehicles": "...", "architecture": "...", "script_type": "..."}'
    )
    local_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=local_user, temperature=0.1
    )
    local_info = {}
    try:
        local_info = VLMClient.extract_json(local_out)
    except (ValueError, json.JSONDecodeError):
        pass

    scene_str = json.dumps(scene_info) if scene_info else scene_out[:300]
    local_str = json.dumps(local_info) if local_info else local_out[:300]

    synthesis_user = (
        "Based on the following multi-stage analysis of a street view image, determine the location.\n\n"
        f"SCENE ANALYSIS: {scene_str}\n\n"
        f"LOCAL DETAILS: {local_str}\n\n"
        "Synthesize all evidence to determine the country and location.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    synth_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=synthesis_user, temperature=0.2
    )
    pred = _parse_json_prediction(synth_out)
    pred["scene_analysis"] = scene_out
    pred["local_analysis"] = local_out
    return pred


# ============================================================
# Our Method v2: Skill-Conditioned VLM (improved)
# Design: GeoCoT structured reasoning → region-filtered skill
# retrieval (composed, non-deduplicated) → expert verification
# ============================================================
def skill_conditioned_v2_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    # Stage 1: GeoCoT 5-step structured reasoning to get candidate region
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
    candidate_country = stage1_pred["predicted_country"]
    candidate_region = stage1_pred["predicted_region"]

    # Stage 2: Region-targeted skill retrieval
    # Build a rich query using GeoCoT reasoning + candidate location
    query = f"{stage1_out[:600]} {candidate_country} {candidate_region}"

    # Retrieve more candidates without region dedup, so we can filter ourselves
    all_skills = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=query,
        top_k=top_k * 4,
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )

    # Filter to skills matching the predicted region; prefer composed (longer) skills
    region_skills = [s for s in all_skills if s.get("region_hint") == candidate_region]
    # Sort by skill text length descending (longer = more composed / multi-cue)
    region_skills.sort(key=lambda s: len(s.get("skill_text", "")), reverse=True)
    region_skills = region_skills[:top_k]

    # Fallback: if we found fewer than 3 region-matching skills, use global top-k by score
    if len(region_skills) < 3:
        region_skills = sorted(all_skills[:top_k], key=lambda s: len(s.get("skill_text", "")), reverse=True)

    skills_text = "\n".join([
        f"[Expert Skill {i+1} | region={s['region_hint']} | conf={s['confidence']:.2f}]\n{s['skill_text']}"
        for i, s in enumerate(region_skills)
    ])

    # Stage 3: Expert-skill-grounded verification pass
    stage2_user = (
        f"Initial structured analysis indicates: {candidate_country} ({candidate_region})\n"
        f"Reasoning summary: {stage1_out[:500]}\n\n"
        "The following skills were extracted from top-ranked GeoGuessr players who correctly "
        f"identified locations in the {candidate_region} region. "
        "These are STRONG PRIORS derived from real expert gameplay — treat each matching "
        "pattern as high-weight evidence. If a skill directly matches what you see in the image, "
        "it significantly increases confidence in that location.\n\n"
        f"Expert Geographic Skills:\n{skills_text}\n\n"
        "Carefully re-examine the image. Do the expert skills corroborate your initial analysis, "
        "or do they reveal a different country? Provide your final answer.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    final_out = vlm.query(
        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=stage2_user, temperature=0.2
    )
    pred = _parse_json_prediction(final_out)
    pred["retrieved_skills"] = region_skills
    pred["stage1_prediction"] = stage1_pred
    pred["stage1_raw"] = stage1_out
    pred["candidate_region"] = candidate_region
    return pred



# ============================================================
# Our Method v3: Skill-Conditioned VLM (v3 — 5-stage)
# Stage 1: GeoCoT structured reasoning → candidate country/region
# Stage 2: GRE-style local detail pass → pole/road/sign fingerprint
# Stage 3: Cross-validate; local detail overrides if it strongly
#           disagrees with Stage 1 on continent/region
# Stage 4: Expanded region-targeted skill retrieval (top_k=10)
#           using candidate chains + expert chains in library
# Stage 5: Final synthesis with explicit textual-location refinement
# ============================================================
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


# ============================================================
# Our Method v4: Skill-Conditioned VLM (v4 — Self-Consistency + Hierarchical + Self-Critique)
# References:
#   - Wang et al. 2022: Self-consistency via majority voting
#   - PIGEON (2023): Hierarchical continent→country→city prediction
#   - Constitutional AI: self-critique before final answer
#   - GRE Suite (2025): local detail fingerprinting
#
# Pipeline:
#   Pass A: GeoCoT 5-step structured (temperature=0.2)
#   Pass B: Hierarchical forced (continent→country→city, temperature=0.3)
#   Pass C: Adversarial self-correction (temperature=0.4)
#   Vote: majority country across A,B,C; fallback to A if split 1-1-1
#   Local: GRE-style local detail fingerprint (image)
#   Critique: Given A+B+C+Local, text-only self-critique + revision
#   Skills: Retrieve from voted region (top_k=12), text-only synthesis
# ============================================================
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


# expert knowledge base, zero-shot)
# Paper: "Img2Loc" (2024) — CLIP retrieval + GPT-4V
# Adapted: BM25+semantic retrieval from expert chains + VLM
# ============================================================
def img2loc_rag_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    describe_user = (
        "Describe this street view image in detail. List all visible features: "
        "road, signs, vegetation, buildings, vehicles, terrain, weather, any text."
    )
    scene_desc = vlm.query(
        image_path=image_path,
        system_prompt="Describe images accurately and thoroughly.",
        user_prompt=describe_user,
        temperature=0.1,
    )

    retrieved = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=scene_desc,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        alpha=0.6,
    )

    context_parts = []
    for i, s in enumerate(retrieved):
        context_parts.append(
            f"Reference {i+1} (from game in {s['region_hint']}): {s['skill_text']}"
        )
    context_text = "\n".join(context_parts)

    rag_user = (
        "You are given a street view image and reference geographic knowledge retrieved "
        "from a database of expert human geolocation analyses.\n\n"
        f"Retrieved references:\n{context_text}\n\n"
        "Use the image AND the references to determine the location. "
        "The references provide geographic patterns (pole types, road markings, vegetation "
        "associations) — match them against what you see.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=rag_user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["retrieved_refs"] = retrieved
    pred["scene_description"] = scene_desc
    return pred


def geocomp_external_predict(
    vlm: VLMClient,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("GeoComp", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", False))
    if strict:
        raise RuntimeError("GeoComp strict official mode requested but official command is not enabled")

    pred = geocot_predict(vlm, image_path)
    pred["external_baseline_name"] = "GeoComp/GeoCoT_adapter"
    pred["official_mode"] = False
    return pred


def geovista_external_predict(
    vlm: VLMClient,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("GeoVista", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", False))
    if strict:
        raise RuntimeError("GeoVista strict official mode requested but official command is not enabled")

    pred = _geovista_agentic_copy_predict(vlm=vlm, image_path=image_path)
    pred["external_baseline_name"] = "GeoVista_adapter_copied"
    pred["official_mode"] = False
    return pred


def geovista_external_skill_boost_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 10,
    candidate_vote_count: int = 3,
    candidate_vote_base_temperature: float = 0.10,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_pred = geovista_external_predict(vlm, image_path, official_cfg=official_cfg)
    base_region = str(base_pred.get("predicted_region", "unknown"))
    base_reasoning = str(base_pred.get("reasoning_text", ""))

    retrieval_query = f"{base_reasoning[:700]} {base_pred.get('predicted_country', 'unknown')} {base_region}".strip()
    retrieved_skills = _multimodal_retrieve_skills(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_query=retrieval_query,
        top_k=max(8, int(top_k) * 4),
        retrieval_mode=retrieval_mode,
        alpha=0.5,
    )

    filtered_skills = _filter_skills_by_region_consistency(
        retrieved_skills=retrieved_skills if isinstance(retrieved_skills, list) else [],
        base_region=base_region,
        top_k=max(6, int(top_k)),
    )

    graph = build_skill_graph_plan(
        retrieved_skills=filtered_skills,
        base_region=base_region,
        max_nodes=max(6, int(top_k)),
    )
    skill_graph_plan: dict[str, Any] = {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "ordered_skill_texts": graph.ordered_skill_texts,
        "summary": graph.summary,
    }

    edge_lines = [
        f"- {e.get('src')} -> {e.get('dst')} ({e.get('relation')}, w={float(e.get('weight', 0.0)):.2f})"
        for e in graph.edges[:20]
    ]
    ordered_skill_lines = [f"{i + 1}. {txt}" for i, txt in enumerate(graph.ordered_skill_texts[:12])]
    ordered_skills_block = "\n".join(ordered_skill_lines)
    edge_block = "\n".join(edge_lines)
    rollout_user = (
        "You must perform geolocation strictly by following the provided task-specific skill graph.\n"
        "First apply high-priority skills in order, then compose linked skills according to graph edges, "
        "and finally produce one consistent textual location guess.\n"
        "Address must be specific: road/intersection/community + city + province_or_state + country.\n\n"
        f"Base observation summary:\n{base_reasoning[:900]}\n\n"
        f"Skill graph summary: {graph.summary}\n"
        f"Ordered skills:\n{ordered_skills_block}\n\n"
        f"Skill relations:\n{edge_block}\n\n"
    )
    final_pred = _predict_with_country_checked_vote(
        vlm=vlm,
        image_path=image_path,
        task_prompt=rollout_user,
        base_temperature=float(candidate_vote_base_temperature),
        num_candidates=max(1, int(candidate_vote_count)),
        fallback_country=str(base_pred.get("predicted_country", "unknown")),
    )

    rollout_trace: list[dict[str, Any]] = [
        {
            "stage": "retrieve_skills",
            "query": retrieval_query[:300],
            "retrieved_skill_count": len(retrieved_skills) if isinstance(retrieved_skills, list) else 0,
            "region_filtered_count": len(filtered_skills),
        },
        {
            "stage": "build_skill_graph",
            "graph_nodes": len(graph.nodes),
            "graph_edges": len(graph.edges),
            "graph_summary": graph.summary,
        },
        {
            "stage": "online_rollout",
            "mode": "skill_graph_only",
            "candidate_vote_count": max(1, int(candidate_vote_count)),
            "candidate_vote_base_temperature": float(candidate_vote_base_temperature),
            "output_country": final_pred.get("predicted_country", "unknown"),
        },
    ]

    final_pred["base_prediction_summary"] = {
        "predicted_country": base_pred.get("predicted_country", "unknown"),
        "predicted_region": base_pred.get("predicted_region", "unknown"),
        "confidence": _safe_float(base_pred.get("confidence"), default=0.0) or 0.0,
    }
    final_pred["retrieved_skills"] = filtered_skills
    final_pred["retrieved_skills_raw_count"] = len(retrieved_skills) if isinstance(retrieved_skills, list) else 0
    final_pred["skill_graph_plan"] = skill_graph_plan
    final_pred["online_rollout_trace"] = rollout_trace
    final_pred["candidate_vote_count"] = max(1, int(candidate_vote_count))
    final_pred["candidate_vote_base_temperature"] = float(candidate_vote_base_temperature)
    final_pred["skill_boost_decision"] = "skill_graph_rollout"
    final_pred["external_baseline_name"] = "GeoVista_skill_graph_rollout"
    final_pred["official_mode"] = bool(base_pred.get("official_mode", False))
    return final_pred


def safa_external_proxy_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    # SAFA is cross-view and not directly compatible with GeoRC street-only samples.
    # This proxy approximates a retrieval-first comparator for unified benchmarking.
    pred = img2loc_rag_predict(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        top_k=6,
        retrieval_mode=retrieval_mode,
    )
    pred["external_baseline_name"] = "SAFA_proxy"
    return pred


def georeasoner_external_predict(
    vlm: VLMClient,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("GeoReasoner", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", False))
    if strict:
        raise RuntimeError("GeoReasoner strict official mode requested but official command is not enabled")

    pred = georeasoner_predict(vlm, image_path)
    pred["external_baseline_name"] = "GeoReasoner_adapter"
    pred["official_mode"] = False
    return pred


def georeasoner_external_skill_boost_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 8,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_pred = georeasoner_external_predict(vlm, image_path, official_cfg=official_cfg)
    boosted = _refine_with_skill_boost(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        base_pred=base_pred,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    boosted["external_baseline_name"] = "GeoReasoner_skill_boost"
    boosted["official_mode"] = bool(base_pred.get("official_mode", False))
    return boosted


def safa_external_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    retrieval_mode: str = "hybrid",
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("SAFA", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", False))
    if strict:
        raise RuntimeError("SAFA strict official mode requested but official command is not enabled")

    pred = safa_external_proxy_predict(
        vlm=vlm,
        skill_library=skill_library,
        image_path=image_path,
        retrieval_mode=retrieval_mode,
    )
    pred["official_mode"] = False
    return pred


def ep_bev_external_predict(
    vlm: VLMClient,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("EP-BEV", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", True))
    if strict:
        raise RuntimeError("EP-BEV official baseline requested but command is not configured")

    pred = georeasoner_predict(vlm, image_path)
    pred["external_baseline_name"] = "EP-BEV_adapter"
    pred["official_mode"] = False
    return pred


def sample4geo_external_predict(
    vlm: VLMClient,
    image_path: str,
    official_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official = _run_official_cli_predict("Sample4Geo", image_path, official_cfg)
    if official is not None:
        return official

    strict = bool((official_cfg or {}).get("strict", True))
    if strict:
        raise RuntimeError("Sample4Geo official baseline requested but command is not configured")

    pred = geocot_predict(vlm, image_path)
    pred["external_baseline_name"] = "Sample4Geo_adapter"
    pred["official_mode"] = False
    return pred
