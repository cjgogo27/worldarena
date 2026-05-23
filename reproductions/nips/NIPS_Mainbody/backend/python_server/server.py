from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio

# Lazy import to avoid hard dependency if not running Python API
try:
    from py_autogeo.orchestrator import run_auto_geo
except Exception:
    run_auto_geo = None

app = FastAPI(title="AutoGeo Python API")

class Scene(BaseModel):
    id: str
    features: dict

class ScenesRequest(BaseModel):
    scenes: list[Scene]

@app.get('/')
def root():
    return {"ok": True, "msg": "AutoGeo Python API server"}

@app.post('/autogeo/run')
async def autogeo_run(payload: ScenesRequest):
    if run_auto_geo is None:
        return {"error": "AutoGeo module not available"}
    scenes = [{"id": s.id, "features": s.features} for s in payload.scenes]
    results = await run_auto_geo(scenes)
    return results

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
