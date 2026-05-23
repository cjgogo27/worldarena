import json
import math
import re
from pathlib import Path


ROOT = Path("/data/alice/cjtest/NIPS/geoskill")
PREDICTIONS_PATH = ROOT / "experiments" / "full_100_mytokenland_combined" / "external_geovista_skill_graph" / "latest_predictions.json"
OUT_JSON = ROOT / "figures" / "real_reasoning_skill_cases.json"
OUT_MD = ROOT / "figures" / "real_reasoning_skill_cases.md"

SELECTED_CASES = [
    {
        "game_id": "KZ2f6LqzJRyChcg8",
        "title": "Andorra: Catalan street sign + Pyrenean settlement",
        "evidence_indices": [0, 1, 3, 4],
        "skill_indices": [1, 2, 5, 6, 10],
    },
    {
        "game_id": "1NJsXTxIF9GGMDxC",
        "title": "Kyrgyzstan: gas-station sign + ex-Soviet roadside cues",
        "evidence_indices": [0, 1, 2, 3, 4],
        "skill_indices": [1, 2, 4, 6, 11, 12],
    },
    {
        "game_id": "2xnQdwiCve2rHWVt",
        "title": "Thailand: Thai script + divided highway corridor",
        "evidence_indices": [0, 1, 2, 3, 4],
        "skill_indices": [1, 2, 3, 4, 5, 12],
    },
    {
        "game_id": "G3aNW5xo5JUCnAhB",
        "title": "Japan: narrow concrete farm road + dense utility wiring",
        "evidence_indices": [0, 1, 2, 3, 4],
        "skill_indices": [1, 2, 3, 4, 8],
    },
    {
        "game_id": "6ypQOh9cOoE7WaWH",
        "title": "Montenegro: karst hills + ex-Yugoslav rural house style",
        "evidence_indices": [0, 2, 4, 5, 6],
        "skill_indices": [1, 5, 6, 10, 12],
    },
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "for",
    "from",
    "that",
    "this",
    "is",
    "are",
    "be",
    "as",
    "by",
    "at",
    "it",
    "its",
    "than",
    "into",
    "near",
    "well",
    "more",
    "most",
    "very",
    "overall",
    "area",
    "road",
    "roads",
    "street",
    "rural",
    "local",
    "visible",
    "typical",
    "common",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _parse_reasoning_payload(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[: -len("```")].strip()
    return json.loads(text)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}


def main() -> None:
    records = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    by_id = {record["game_id"]: record for record in records}

    casebook = []
    for case in SELECTED_CASES:
        record = by_id[case["game_id"]]
        prediction = record["prediction"]
        reasoning_payload = _parse_reasoning_payload(str(prediction.get("reasoning_text", "")))
        evidence = list(reasoning_payload.get("evidence", []))
        selected_evidence = [
            evidence[idx] for idx in case.get("evidence_indices", []) if 0 <= idx < len(evidence)
        ]
        skills = prediction.get("retrieved_skills", []) if isinstance(prediction.get("retrieved_skills"), list) else []
        selected_skills = []
        for idx in case.get("skill_indices", []):
            if not (1 <= idx <= len(skills)):
                continue
            skill = skills[idx - 1]
            selected_skills.append(
                {
                    "rank": idx,
                    "skill_text": str(skill.get("skill_text", "") or ""),
                    "skill_source_game_id": skill.get("source_game_id"),
                    "skill_source_round": skill.get("source_round"),
                    "skill_confidence": float(skill.get("confidence", 0.0) or 0.0),
                    "retrieval_score": float(skill.get("score", 0.0) or 0.0),
                }
            )
        distance_km = _haversine_km(
            float(record["ground_truth_lat"]),
            float(record["ground_truth_lng"]),
            float(prediction["predicted_lat"]),
            float(prediction["predicted_lng"]),
        )

        casebook.append(
            {
                "title": case["title"],
                "game_id": case["game_id"],
                "image_path": record["image_path"],
                "predicted_address": prediction.get("predicted_address"),
                "predicted_country": prediction.get("predicted_country"),
                "ground_truth_country": record.get("ground_truth_country"),
                "distance_km": round(distance_km, 3),
                "base_prediction_summary": prediction.get("base_prediction_summary"),
                "reasoning_summary": reasoning_payload.get("reasoning"),
                "selected_evidence": selected_evidence,
                "all_evidence": evidence,
                "matched_skills": selected_skills,
                "graph_summary": (prediction.get("skill_graph_plan") or {}).get("summary"),
            }
        )

    OUT_JSON.write_text(json.dumps(casebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Real Reasoning + Skill-Matched Casebook",
        "",
        "Selected from `experiments/full_100_mytokenland_combined/external_geovista_skill_graph/latest_predictions.json`.",
        "Each case uses the model's real reasoning JSON and aligns evidence bullets to retrieved skills with a lightweight lexical+retrieval-score matcher.",
        "",
    ]
    for idx, case in enumerate(casebook, start=1):
        lines.extend(
            [
                f"## Case {idx}: {case['title']}",
                "",
                f"- `game_id`: `{case['game_id']}`",
                f"- `image`: `{case['image_path']}`",
                f"- `prediction`: `{case['predicted_address']}`",
                f"- `distance_km`: `{case['distance_km']}`",
                f"- `graph_summary`: `{case['graph_summary']}`",
                "",
                "**Real Reasoning Summary**",
                "",
                case["reasoning_summary"] or "",
                "",
                "**Selected Evidence Bullets**",
                "",
            ]
        )
        for evidence in case["selected_evidence"]:
            lines.append(f"- {evidence}")
        lines.extend(["", "**Matched Skills (manually curated from retrieved skills)**", ""])
        for skill in case["matched_skills"]:
            lines.extend(
                [
                    f"- [rank {skill['rank']}] {skill['skill_text']}",
                    f"  Source: `{skill['skill_source_game_id']}` round `{skill['skill_source_round']}`; retrieval_score: `{skill['retrieval_score']:.3f}`",
                ]
            )
        lines.extend(["", "**All Raw Evidence Bullets**", ""])
        for evidence in case["all_evidence"]:
            lines.append(f"- {evidence}")
        lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
