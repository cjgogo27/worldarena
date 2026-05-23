# PyAutoGeo: Python Prototype for Multi-Image Joint Geolocalization

Overview
- A Python-based prototype implementing a Master-Agent + SubAgents pattern for multi-image geolocalization tasks.
- Uses mock tools to simulate vision, map, and knowledge-based reasoning.

Quickstart
- Install dependencies (in a clean venv):
  - pip install fastapi uvicorn pytest
- Run the Python API server (optional):
  - python backend/python_server/server.py
- Run the demo:
  - python py_autogeo/demo.py
- Run tests:
  - pytest tests/test_py_autogeo.py

Project structure
- py_autogeo/           Python AutoGeo prototype (master + subagents)
- backend/python_server/  FastAPI-based Python REST API for AutoGeo
- tests/                Tests for the Python prototype
- autogeo/               Legacy JS AutoGeo (reference)

What to replace for real usage
- Swap mock tools in py_autogeo/tools.py with real services (actual vision, OCR, GIS APIs).
- Enhance SubAgent scheduling and Master-Agent coordination to reflect your research design (staged communication, belief exchange).
- Add persistence, authentication, and robust error handling for production use.

Notes
- PDF input note: Current environment cannot read PDFs directly. Provide text extracts or summaries for alignment.
