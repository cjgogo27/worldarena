import json

import math

import re

from typing import Any



from .skill_library import SkillLibrary

from .skill_parser import COUNTRY_NAME_TO_ISO2, COUNTRY_TO_REGION

from .vlm_client import VLMClient



_JSON_SCHEMA_INSTRUCTION = """You MUST respond with ONLY a valid JSON object in the following format (no markdown, no extra text):
{
  "country": "<full country name>",
  "country_code": "<ISO 3166-1 alpha-2 lowercase code>",
  "region": "<continent or sub-region>",
  "lat": <latitude as decimal number>,
  "lng": <longitude as decimal number>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<your step-by-step reasoning>",
  "evidence": ["<visual cue 1>", "<visual cue 2>", ...]
}"""



_GEOLOCATION_SYSTEM = (

    "You are an expert geolocation analyst. You analyze street view images to determine "

    "the precise location. Use visual evidence: road markings, signs, vegetation, architecture, "

    "utility poles, driving side, license plates, terrain, climate indicators. "

    "Always provide your best guess even when uncertain — never refuse to answer."

)





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

    try:

        parsed = VLMClient.extract_json(raw_text)

    except (ValueError, json.JSONDecodeError):

        pass



    if parsed and isinstance(parsed, dict):

        country_code = str(parsed.get("country_code", "")).lower().strip()

        country_name = str(parsed.get("country", "")).lower().strip()

        if not country_code or country_code not in COUNTRY_TO_REGION:

            country_code = _country_name_to_iso(country_name)

        lat = _safe_float(parsed.get("lat"))

        lng = _safe_float(parsed.get("lng"))

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

        conf = _extract_confidence_from_text(raw_text)

        reasoning = raw_text

        evidence = _extract_evidence_spans(raw_text)



    region = COUNTRY_TO_REGION.get(country_code, "unknown")

    return {

        "predicted_country": country_code,

        "predicted_region": region,

        "predicted_lat": lat if lat is not None else math.nan,

        "predicted_lng": lng if lng is not None else math.nan,

        "reasoning_text": raw_text,

        "evidence_spans": evidence,

        "confidence": max(0.0, min(1.0, conf if conf is not None else 0.5)),

    }





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











def direct_vlm_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:

    user = (

        "Analyze this street view image and identify the country and approximate location.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)

    return _parse_json_prediction(out)











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

        "Now determine the exact country and approximate coordinates using fine-grained visual evidence "

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

        "STEP 5 - COORDINATE ESTIMATION:\n"

        "Based on all evidence, estimate latitude and longitude.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)

    return _parse_json_prediction(out)













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















def skill_conditioned_v2_predict(

    vlm: VLMClient,

    skill_library: SkillLibrary,

    image_path: str,

    top_k: int = 8,

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

        "STEP 5 - COORDINATE ESTIMATION:\n"

        "Based on all evidence, estimate latitude and longitude.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    stage1_out = vlm.query(

        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=stage1_user, temperature=0.2

    )

    stage1_pred = _parse_json_prediction(stage1_out)

    candidate_country = stage1_pred["predicted_country"]

    candidate_region = stage1_pred["predicted_region"]







    query = f"{stage1_out[:600]} {candidate_country} {candidate_region}"





    all_skills = _multimodal_retrieve_skills(

        vlm=vlm,

        skill_library=skill_library,

        image_path=image_path,

        base_query=query,

        top_k=top_k * 4,

        retrieval_mode=retrieval_mode,

        alpha=0.5,

    )





    region_skills = [s for s in all_skills if s.get("region_hint") == candidate_region]



    region_skills.sort(key=lambda s: len(s.get("skill_text", "")), reverse=True)

    region_skills = region_skills[:top_k]





    if len(region_skills) < 3:

        region_skills = sorted(all_skills[:top_k], key=lambda s: len(s.get("skill_text", "")), reverse=True)



    skills_text = "\n".join([

        f"[Expert Skill {i+1} | region={s['region_hint']} | conf={s['confidence']:.2f}]\n{s['skill_text']}"

        for i, s in enumerate(region_skills)

    ])





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

        "STEP 5 - COORDINATE ESTIMATION:\n"

        "Based on all evidence, estimate latitude and longitude.\n\n"

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

        "Synthesize all evidence. Use local details to pinpoint coordinates precisely.\n\n"

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



    pass_a_user = (

        "You are an expert GeoGuessr player. Analyze this street view image using the "

        "systematic reasoning strategy that top human players use:\n\n"

        "STEP 1 - HEMISPHERE & CLIMATE ZONE: Look at sun position/shadows, assess climate.\n\n"

        "STEP 2 - CONTINENT NARROWING: Use driving side, road quality, landscape.\n\n"

        "STEP 3 - COUNTRY IDENTIFICATION: Focus on utility poles, bollards, road markings, "

        "sign design, license plate color/shape, Google car type.\n\n"

        "STEP 4 - REGION WITHIN COUNTRY: Vegetation, terrain, architecture, visible text.\n\n"

        "STEP 5 - COORDINATE ESTIMATION: Estimate latitude and longitude.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    pass_a_out = vlm.query(

        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_a_user, temperature=0.2

    )

    pred_a = _parse_json_prediction(pass_a_out)





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

        "LEVEL 4 — COORDINATES: Based on levels 1-3, estimate lat/lng precisely.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    pass_b_out = vlm.query(

        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_b_user, temperature=0.3

    )

    pred_b = _parse_json_prediction(pass_b_out)







    pass_c_user = (

        "You are an expert GeoGuessr player tasked with identifying a location from this street view image.\n\n"

        "IMPORTANT: Many locations are deliberately chosen to be ambiguous and look like other countries. "

        "Before committing to an answer, explicitly consider and RULE OUT the most visually similar alternatives:\n\n"

        "1. First, note the single most distinctive feature in the image (the 'smoking gun' clue).\n"

        "2. List the top 3 candidate countries this could be, with the key evidence FOR each.\n"

        "3. For each candidate, identify the SINGLE piece of evidence that would RULE IT OUT.\n"

        "4. Apply this elimination: which candidates are ruled out and why?\n"

        "5. Commit to your final answer with precise coordinates.\n\n"

        + _JSON_SCHEMA_INSTRUCTION

    )

    pass_c_out = vlm.query(

        image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=pass_c_user, temperature=0.4

    )

    pred_c = _parse_json_prediction(pass_c_out)





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





    votes: dict[str, list[int]] = {}

    for i, pred in enumerate([pred_a, pred_b, pred_c]):

        cc = pred["predicted_country"]

        if cc and cc != "unknown":

            votes.setdefault(cc, []).append(i)





    voted_country = "unknown"

    voted_region = "unknown"

    for cc, indices in sorted(votes.items(), key=lambda kv: -len(kv[1])):

        if len(indices) >= 2:

            voted_country = cc

            voted_region = COUNTRY_TO_REGION.get(cc, "unknown")

            break





    if voted_country == "unknown":

        local_implied_country = str(local_info.get("implied_country", "")).lower().strip()

        local_iso = _country_name_to_iso(local_implied_country) if local_implied_country else "unknown"

        if local_iso != "unknown":

            voted_country = local_iso

            voted_region = COUNTRY_TO_REGION.get(local_iso, "unknown")

        else:

            voted_country = pred_a["predicted_country"]

            voted_region = pred_a["predicted_region"]





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







    vote_summary = ", ".join([f"Pass {'ABC'[i]}→{pred['predicted_country']}"

                               for i, pred in enumerate([pred_a, pred_b, pred_c])])

    all_agree = len(set(p["predicted_country"] for p in [pred_a, pred_b, pred_c])) == 1

    agreement_note = (

        f"All 3 passes agree: {voted_country}. High confidence." if all_agree

        else f"Vote: {vote_summary}. Majority: {voted_country}."

    )





    agreeing_preds = [p for p in [pred_a, pred_b, pred_c]

                      if p["predicted_country"] == voted_country

                      and not math.isnan(p.get("predicted_lat", math.nan))

                      and not math.isnan(p.get("predicted_lng", math.nan))]

    avg_lat = sum(p["predicted_lat"] for p in agreeing_preds) / len(agreeing_preds) if agreeing_preds else math.nan

    avg_lng = sum(p["predicted_lng"] for p in agreeing_preds) / len(agreeing_preds) if agreeing_preds else math.nan





    pass_summaries = (

        f"Pass A (GeoCoT): country={pred_a['predicted_country']}, "

        f"coords=({pred_a['predicted_lat']:.2f},{pred_a['predicted_lng']:.2f}), "

        f"reasoning={pass_a_out[:250]}\n\n"

        f"Pass B (Hierarchical): country={pred_b['predicted_country']}, "

        f"coords=({pred_b['predicted_lat']:.2f},{pred_b['predicted_lng']:.2f}), "

        f"reasoning={pass_b_out[:200]}\n\n"

        f"Pass C (Adversarial): country={pred_c['predicted_country']}, "

        f"coords=({pred_c['predicted_lat']:.2f},{pred_c['predicted_lng']:.2f}), "

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

        "4. What is your final refined answer with the most precise coordinates possible?\n\n"

        "IMPORTANT: The average coordinates from agreeing passes are "

        f"lat={avg_lat:.4f}, lng={avg_lng:.4f} — use this as your coordinate baseline "

        "and refine based on the local detail and expert skills.\n\n"

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

            f"Average coords from agreeing passes: lat={avg_lat:.4f}, lng={avg_lng:.4f}\n"

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





def geocomp_external_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:



    pred = geocot_predict(vlm, image_path)

    pred["external_baseline_name"] = "GeoComp/GeoCoT"

    return pred





def geovista_external_predict(vlm: VLMClient, image_path: str) -> dict[str, Any]:



    stage1_user = (

        "You are GeoVista-like agent. First output a concise scene plan in JSON: "

        "{\"region_hypothesis\": str, \"country_candidates\": [str], \"visual_focus\": [str]}."

    )

    stage1_out = vlm.query(

        image_path=image_path,

        system_prompt=_GEOLOCATION_SYSTEM,

        user_prompt=stage1_user,

        temperature=0.2,

    )

    stage2_user = (

        f"Agent plan: {stage1_out[:500]}\n"

        "Now provide final geolocation JSON using the required schema."

        + _JSON_SCHEMA_INSTRUCTION

    )

    out = vlm.query(

        image_path=image_path,

        system_prompt=_GEOLOCATION_SYSTEM,

        user_prompt=stage2_user,

        temperature=0.2,

    )

    pred = _parse_json_prediction(out)

    pred["external_baseline_name"] = "GeoVista"

    pred["agent_plan"] = stage1_out

    return pred





def safa_external_proxy_predict(

    vlm: VLMClient,

    skill_library: SkillLibrary,

    image_path: str,

    retrieval_mode: str = "hybrid",

) -> dict[str, Any]:





    pred = img2loc_rag_predict(

        vlm=vlm,

        skill_library=skill_library,

        image_path=image_path,

        top_k=6,

        retrieval_mode=retrieval_mode,

    )

    pred["external_baseline_name"] = "SAFA_proxy"

    return pred
