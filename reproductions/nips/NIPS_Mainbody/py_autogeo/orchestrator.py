import asyncio
from typing import List, Dict, Any
from .subagent import SubAgent

async def _run_all_agents(scenes: List[Dict[str, Any]]):
    agent_vision = SubAgent(1, 'vision', ['zoom', 'ocr', 'retrieval'])
    agent_maps = SubAgent(2, 'maps', ['poi_keyword', 'poi_detail', 'static_map'])
    agent_knowledge = SubAgent(3, 'knowledge', ['geo_info', 'satellite_map'])

    results_vision = asyncio.create_task(agent_vision.run(scenes))
    results_maps = asyncio.create_task(agent_maps.run(scenes))
    results_knowledge = asyncio.create_task(agent_knowledge.run(scenes))

    res = await asyncio.gather(results_vision, results_maps, results_knowledge)
    return res  # list of three lists: [vision, maps, knowledge]

def _aggregate(scenes: List[Dict[str, Any]], per_agent: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # per_agent is [vision_results, maps_results, knowledge_results]
    scene_map = {}
    for agent_idx, agen in enumerate(per_agent):
        for item in agen:
            sid = item.get('scene_id')
            if sid not in scene_map:
                scene_map[sid] = []
            scene_map[sid].append({"agent": item.get('agent'), "city": item.get('city'), "conf": item.get('confidence')})
    results = []
    for scene in scenes:
        sid = scene.get('id')
        entries = scene_map.get(sid, [])
        # compute best city by average confidence
        city_scores = {}
        for e in entries:
            c = e['city']
            if c not in city_scores:
                city_scores[c] = []
            city_scores[c].append(e['conf'])
        best_city = 'Unknown'
        best_avg = 0.0
        for c, confs in city_scores.items():
            if len(confs) == 0:
                continue
            avg = sum(confs) / len(confs)
            if avg > best_avg:
                best_avg = avg
                best_city = c
        results.append({"scene_id": sid, "city": best_city, "confidence": best_avg, "per_agent": entries})
    return results

async def run_auto_geo(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_agent = _run_all_agents(scenes)
    vision, maps, knowledge = await per_agent
    # aggregate
    results = _aggregate(scenes, [vision, maps, knowledge])
    return results
