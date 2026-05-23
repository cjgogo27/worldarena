import json

from collections import defaultdict

from pathlib import Path

from typing import Any



from .skill_parser import COUNTRY_TO_REGION, Skill

from .vlm_client import VLMClient





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

        "- no markdown\n\n"

        f"Failed cases:\n{json.dumps(packed, ensure_ascii=False)}"

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

            if rh not in {"europe", "asia", "north_america", "south_america", "africa", "oceania", "unknown"}:

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





    dedup = {s.skill_text: s for s in fused}

    return list(dedup.values())
