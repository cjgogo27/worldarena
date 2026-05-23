AutoGeo for Python: Master-Agent + SubAgents Prototype

Overview
- A compact Python-based prototype implementing a Master-Agent and SubAgents for multi-image geolocalization tasks.
- Uses mock tools to simulate vision, map, and knowledge-based reasoning.

Usage
- Run the demo: python -m py_autogeo.demo
- Run programmatically: from py_autogeo.orchestrator import run_auto_geo; await run_auto_geo([...])

Extending
- Replace tools in py_autogeo/tools.py with real services (OCR, image analysis, maps, etc.).
- Enhance SubAgent scheduling and Master-Agent coordination to reflect your research design (e.g., staged communication, belief exchange).
- Add tests in tests/ to validate behavior with various scene inputs.
