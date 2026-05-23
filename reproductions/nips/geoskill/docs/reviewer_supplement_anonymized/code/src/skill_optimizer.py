import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .skill_parser import COUNTRY_TO_REGION, Skill
from .vlm_client import VLMClient
from .web_search import search_web


_ALLOWED_REGION_HINTS = {"europe", "asia", "north_america", "south_america", "africa", "oceania", "unknown"}


def _build_failure_search_queries(case: dict[str, Any], limit: int = 2) -> list[str]:
    gt_country = str(case.get("gt_country", "unknown")).strip().lower()
    pred_country = str(case.get("pred_country", "unknown")).strip().lower()
    reasoning = str(case.get("reasoning", "")).replace("\n", " ").strip()
    reasoning = " ".join(reasoning.split())[:220]

    queries = [
        (
            f"{gt_country} street view geolocation cues road markings utility poles "
            f"driving side traffic signs"
        ),
    ]
    if pred_country and pred_country != "unknown" and pred_country != gt_country:
        queries.append(
            f"{gt_country} vs {pred_country} geolocation differences street view clues"
        )
    if reasoning:
        queries.append(
            f"street view geolocation common mistakes {gt_country} {reasoning}"
        )

    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        qq = " ".join(str(q).split())
        if not qq or qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
        if len(out) >= max(1, limit):
            break
    return out


def _collect_web_refs_for_failures(
    packed_cases: list[dict[str, Any]],
    provider: str,
    api_key: str,
    search_max_cases: int,
    search_queries_per_case: int,
    search_results_per_query: int,
    timeout_seconds: float,
    char_budget: int,
) -> list[dict[str, str]]:
    if not packed_cases:
        return []

    refs: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    remaining = max(200, int(char_budget))

    for case in packed_cases[: max(1, int(search_max_cases))]:
        queries = _build_failure_search_queries(case, limit=search_queries_per_case)
        for q in queries:
            docs = search_web(
                query=q,
                provider=provider,
                api_key=api_key,
                count=search_results_per_query,
                timeout_seconds=timeout_seconds,
            )
            for d in docs:
                url = str(d.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                item = {
                    "query": q,
                    "title": str(d.get("title", "")).strip(),
                    "snippet": str(d.get("snippet", "")).strip(),
                    "url": url,
                }
                delta = len(json.dumps(item, ensure_ascii=False))
                if delta > remaining:
                    return refs
                refs.append(item)
                remaining -= delta

    return refs


def dump_skills_jsonl(skills: list[Skill], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in skills:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_skills_jsonl(path: Path) -> list[Skill]:
    if not path.exists():
        return []
    loaded: list[Skill] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            loaded.append(
                Skill(
                    skill_text=str(obj.get("skill_text", "")).strip(),
                    region_hint=str(obj.get("region_hint", "unknown")),
                    confidence=float(obj.get("confidence", 0.65)),
                    visual_cues=[str(x) for x in obj.get("visual_cues", [])],
                    source_game_id=str(obj.get("source_game_id", "generated")),
                    source_round=int(obj.get("source_round", 0)),
                )
            )
    return [s for s in loaded if s.skill_text]


def synthesize_failure_skills(
    vlm: VLMClient,
    failed_records: list[dict[str, Any]],
    max_records: int = 20,
    skill_search_enabled: bool = False,
    skill_search_provider: str = "brave",
    skill_search_api_key: str = "",
    skill_search_max_cases: int = 4,
    skill_search_queries_per_case: int = 2,
    skill_search_results_per_query: int = 3,
    skill_search_timeout_seconds: float = 8.0,
    skill_search_context_char_budget: int = 3000,
) -> list[Skill]:
    if not failed_records:
        return []

    sampled = failed_records[:max_records]
    packed = []
    for r in sampled:
        pred = r.get("prediction", {})
        gt_country = str(r.get("ground_truth_country", "unknown"))
        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")
        packed.append(
            {
                "game_id": r.get("game_id"),
                "gt_country": gt_country,
                "gt_region": gt_region,
                "pred_country": pred.get("predicted_country", "unknown"),
                "pred_region": pred.get("predicted_region", "unknown"),
                "reasoning": str(pred.get("reasoning_text", ""))[:700],
            }
        )

    web_refs: list[dict[str, str]] = []
    if skill_search_enabled:
        web_refs = _collect_web_refs_for_failures(
            packed_cases=packed,
            provider=skill_search_provider,
            api_key=skill_search_api_key,
            search_max_cases=skill_search_max_cases,
            search_queries_per_case=skill_search_queries_per_case,
            search_results_per_query=skill_search_results_per_query,
            timeout_seconds=skill_search_timeout_seconds,
            char_budget=skill_search_context_char_budget,
        )

    user_prompt = (
        "You are a geolocation skill optimizer.\n"
        "Given failed geolocation cases, extract reusable skills that reduce these failure patterns.\n"
        "Return ONLY JSON array. Each element:\n"
        "{\"skill_text\": str, \"region_hint\": str, \"confidence\": float, \"visual_cues\": [str]}\n"
        "Constraints:\n"
        "- skill_text should be concise and actionable\n"
        "- include both atomic and composed skills\n"
        "- region_hint one of: europe, asia, north_america, south_america, africa, oceania, unknown\n"
        "- confidence in [0,1]\n"
        "- prioritize robust and repeatedly-supported cues\n"
        "- no markdown\n\n"
        f"Failed cases:\n{json.dumps(packed, ensure_ascii=False)}"
    )
    if web_refs:
        user_prompt += (
            "\n\nExternal references from web search (noisy, use cautiously):\n"
            f"{json.dumps(web_refs, ensure_ascii=False)}\n"
            "Use web references only as weak evidence. Keep only cues supported by multiple cases or standard geography knowledge."
        )

    raw = vlm.query(
        image_path=None,
        system_prompt="You produce strict JSON only.",
        user_prompt=user_prompt,
        temperature=0.1,
    )

    items: list[dict[str, Any]] = []
    try:
        parsed = VLMClient.extract_json(raw)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("skills"), list):
            items = parsed["skills"]
    except Exception:
        return []

    out: list[Skill] = []
    for i, x in enumerate(items):
        try:
            st = str(x.get("skill_text", "")).strip()
            if not st:
                continue
            rh = str(x.get("region_hint", "unknown")).strip().lower()
            if rh not in _ALLOWED_REGION_HINTS:
                rh = "unknown"
            c = float(x.get("confidence", 0.65))
            c = max(0.0, min(1.0, c))
            cues = [str(v).lower() for v in x.get("visual_cues", [])][:8]
            out.append(
                Skill(
                    skill_text=st,
                    region_hint=rh,
                    confidence=c,
                    visual_cues=cues,
                    source_game_id=f"failure_recovery_{i}",
                    source_round=0,
                )
            )
        except Exception:
            continue
    return out


def fuse_atomic_skills(skills: list[Skill], min_group_size: int = 2) -> list[Skill]:
    if not skills:
        return []

    grouped: dict[tuple[str, str], list[Skill]] = defaultdict(list)
    for s in skills:
        if not s.visual_cues:
            continue
        for cue in s.visual_cues[:3]:
            grouped[(s.region_hint, cue)].append(s)

    fused: list[Skill] = []
    for (region, cue), group in grouped.items():
        uniq = {g.skill_text: g for g in group}
        vals = list(uniq.values())
        if len(vals) < min_group_size:
            continue

        top = sorted(vals, key=lambda x: -x.confidence)[:3]
        snippets = [t.skill_text for t in top]
        fused_text = (
            f"Composed skill: if cue '{cue}' co-occurs with {', '.join(top[0].visual_cues[:2])}, "
            f"prioritize {region}. Supporting patterns: " + " | ".join(snippets)
        )
        conf = min(0.95, sum(t.confidence for t in top) / max(1, len(top)) + 0.08)
        cues = sorted({v for t in top for v in t.visual_cues})[:10]
        fused.append(
            Skill(
                skill_text=fused_text,
                region_hint=region,
                confidence=conf,
                visual_cues=cues,
                source_game_id="skill_fusion",
                source_round=0,
            )
        )

    # Deduplicate by text
    dedup = {s.skill_text: s for s in fused}
    return list(dedup.values())
