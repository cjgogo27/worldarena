import asyncio
from typing import List, Dict, Any, Optional
from . import tools as tools_mod

class SubAgent:
    def __init__(self, agent_id: int, role: str, tool_names: List[str]):
        self.id = agent_id
        self.role = role
        self.tool_names = tool_names
        # map tool name to actual callable
        self.tool_map = {
            'zoom': tools_mod.zoom_tool,
            'ocr': tools_mod.ocr_tool,
            'retrieval': tools_mod.image_retrieval_tool,
            'poi_keyword': tools_mod.poi_keyword_search_tool,
            'poi_detail': tools_mod.poi_detail_query_tool,
            'static_map': tools_mod.static_map_query_tool,
            'satellite_map': tools_mod.satellite_map_query_tool,
            'geo_info': tools_mod.geoinformation_search_tool,
        }

    async def _call_tool(self, tool_name: str, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fn = self.tool_map.get(tool_name)
        if not fn:
            return None
        # Tools are synchronous in this prototype; wrap in a coroutine for uniform async API
        return fn(scene)

    async def run_on_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        best_city = None
        best_conf = -1.0
        for tn in self.tool_names:
            out = await self._call_tool(tn, scene)
            if not out:
                continue
            city = out.get('city')
            conf = out.get('confidence', 0.0)
            if city and conf > best_conf:
                best_city, best_conf = city, conf
        if best_city is None:
            best_city, best_conf = 'Unknown', 0.0
        return {"city": best_city, "confidence": float(best_conf), "agent": self.id}

    async def run(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for scene in scenes:
            r = await self.run_on_scene(scene)
            results.append({"scene_id": scene.get('id'), **r})
        return results
