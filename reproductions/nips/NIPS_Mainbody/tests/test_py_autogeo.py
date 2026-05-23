import asyncio
import json

from py_autogeo.orchestrator import run_auto_geo

async def _run_demo():
    scenes = [
        { 'id': 's1', 'features': { 'landmarks': ['Eiffel Tower'], 'country': 'France' } },
        { 'id': 's2', 'features': { 'landmarks': [], 'country': 'France', 'signText': 'Rue de la Paix' } },
    ]
    return await run_auto_geo(scenes)

def test_run_auto_geo_basic():
    results = asyncio.get_event_loop().run_until_complete(_run_demo())
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert 'scene_id' in r or 'sceneId' in r
        assert 'city' in r
