import asyncio
from .orchestrator import run_auto_geo

async def main():
    scenes = [
        { 'id': 's1', 'features': { 'landmarks': ['Eiffel Tower'], 'country': 'France', 'signText': '' } },
        { 'id': 's2', 'features': { 'landmarks': [], 'country': 'France', 'signText': 'Rue de la Paix' } },
        { 'id': 's3', 'features': { 'landmarks': ['Golden Gate Bridge'], 'country': 'USA', 'signText': '' } },
    ]
    results = await run_auto_geo(scenes)
    print("AutoGeo Python Demo Results:")
    import json
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
