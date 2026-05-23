from __future__ import annotations

from typing import Any


def run_search(query: str, retriever_name: str = "tavily") -> list[dict[str, Any]]:
    """Fallback web search shim used when upstream gpt_researcher is unavailable.

    Returns a minimal non-empty result so GeoVista's agent loop can continue
    without hard failing on optional web search dependency.
    """
    q = (query or "").strip()
    return [
        {
            "title": "search_unavailable_fallback",
            "content": f"Web search backend is unavailable in this runtime. Query: {q}",
            "url": "",
            "source": retriever_name,
        }
    ]
