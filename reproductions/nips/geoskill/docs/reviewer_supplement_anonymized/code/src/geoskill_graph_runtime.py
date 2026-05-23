from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass
class SkillGraphPlan:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    ordered_skill_texts: list[str]
    summary: str


def build_skill_graph_plan(
    retrieved_skills: list[dict[str, Any]],
    base_region: str,
    max_nodes: int = 8,
) -> SkillGraphPlan:
    if not retrieved_skills:
        return SkillGraphPlan(nodes=[], edges=[], ordered_skill_texts=[], summary="no-skills")

    picked = [s for s in retrieved_skills if isinstance(s, dict)][: max(1, int(max_nodes))]
    picked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    nodes: list[dict[str, Any]] = []
    for i, s in enumerate(picked):
        nodes.append(
            {
                "id": f"skill_{i+1}",
                "skill_text": str(s.get("skill_text", "")).strip(),
                "region_hint": str(s.get("region_hint", "unknown")).strip().lower(),
                "confidence": float(s.get("confidence", 0.0) or 0.0),
                "score": float(s.get("score", 0.0) or 0.0),
                "visual_cues": [str(v) for v in s.get("visual_cues", [])][:6],
            }
        )

    # Build sparse compositional edges based on region agreement and cue overlap.
    edges: list[dict[str, Any]] = []
    for a, b in combinations(nodes, 2):
        a_region = a.get("region_hint", "unknown")
        b_region = b.get("region_hint", "unknown")
        cues_a = set(a.get("visual_cues", []))
        cues_b = set(b.get("visual_cues", []))
        overlap = sorted(cues_a.intersection(cues_b))[:3]

        if a_region == b_region and a_region != "unknown":
            edges.append(
                {
                    "src": a["id"],
                    "dst": b["id"],
                    "relation": "region_support",
                    "weight": 1.0,
                    "shared_cues": overlap,
                }
            )
        elif overlap:
            edges.append(
                {
                    "src": a["id"],
                    "dst": b["id"],
                    "relation": "cue_composition",
                    "weight": 0.6,
                    "shared_cues": overlap,
                }
            )

    ordered = [n["skill_text"] for n in nodes if n.get("skill_text")]
    region_hist: dict[str, int] = {}
    for n in nodes:
        r = str(n.get("region_hint", "unknown"))
        region_hist[r] = region_hist.get(r, 0) + 1

    top_region = max(region_hist.items(), key=lambda kv: kv[1])[0] if region_hist else "unknown"
    summary = (
        f"nodes={len(nodes)}, edges={len(edges)}, base_region={base_region or 'unknown'}, "
        f"top_skill_region={top_region}"
    )
    return SkillGraphPlan(nodes=nodes, edges=edges, ordered_skill_texts=ordered, summary=summary)


def summarize_rollout_skill_edges(records: list[dict[str, Any]], top_k: int = 50) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}

    for rec in records:
        pred = rec.get("prediction", {}) if isinstance(rec, dict) else {}
        trace = pred.get("skill_graph_plan", {}) if isinstance(pred, dict) else {}
        nodes = trace.get("nodes", []) if isinstance(trace, dict) else []
        texts = [str(n.get("skill_text", "")).strip() for n in nodes if isinstance(n, dict)]
        texts = [t for t in texts if t]
        uniq = sorted(set(texts))
        for a, b in combinations(uniq, 2):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    return [
        {
            "skill_a": a,
            "skill_b": b,
            "co_occurrence": c,
        }
        for (a, b), c in ranked
    ]
