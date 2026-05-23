import json
import gzip
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_WIKIPEDIA_OPENSEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"


def _safe_text(value: Any, max_chars: int = 320) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def brave_web_search(
    query: str,
    api_key: str,
    count: int = 5,
    timeout_seconds: float = 10.0,
) -> list[dict[str, str]]:
    q = str(query or "").strip()
    if not q or not api_key:
        return []

    params = urlencode({"q": q, "count": max(1, min(int(count), 20))})
    url = f"{_BRAVE_ENDPOINT}?{params}"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
            "User-Agent": "geoskill-search-evolution/1.0",
        },
    )

    try:
        with urlopen(req, timeout=max(1.0, float(timeout_seconds))) as resp:
            raw = resp.read()
            encoding = str(resp.headers.get("Content-Encoding", "")).lower()
            if "gzip" in encoding:
                raw = gzip.decompress(raw)
            body = raw.decode("utf-8", errors="ignore")
            payload = json.loads(body)
    except Exception:
        return []

    results = payload.get("web", {}).get("results", [])
    if not isinstance(results, list):
        return []

    docs: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        url_val = _safe_text(item.get("url"), max_chars=500)
        if not url_val or url_val in seen_urls:
            continue
        seen_urls.add(url_val)

        desc = _safe_text(item.get("description"), max_chars=420)
        extra = item.get("extra_snippets", [])
        if isinstance(extra, list) and extra:
            extra_joined = " ".join(str(x) for x in extra[:2])
            if extra_joined:
                desc = _safe_text(f"{desc} {extra_joined}".strip(), max_chars=420)

        docs.append(
            {
                "title": _safe_text(item.get("title"), max_chars=180),
                "url": url_val,
                "snippet": desc,
            }
        )

    return docs


def wikipedia_opensearch(
    query: str,
    count: int = 5,
    timeout_seconds: float = 10.0,
) -> list[dict[str, str]]:
    q = str(query or "").strip()
    if not q:
        return []

    params = urlencode(
        {
            "action": "opensearch",
            "search": q,
            "limit": max(1, min(int(count), 20)),
            "namespace": 0,
            "format": "json",
        }
    )
    url = f"{_WIKIPEDIA_OPENSEARCH_ENDPOINT}?{params}"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "geoskill-search-evolution/1.0",
        },
    )

    try:
        with urlopen(req, timeout=max(1.0, float(timeout_seconds))) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            payload = json.loads(body)
    except Exception:
        return []

    if not isinstance(payload, list) or len(payload) < 4:
        return []

    titles = payload[1] if isinstance(payload[1], list) else []
    snippets = payload[2] if isinstance(payload[2], list) else []
    urls = payload[3] if isinstance(payload[3], list) else []

    docs: list[dict[str, str]] = []
    for i, title in enumerate(titles):
        url_val = _safe_text(urls[i] if i < len(urls) else "", max_chars=500)
        if not url_val:
            continue
        docs.append(
            {
                "title": _safe_text(title, max_chars=180),
                "url": url_val,
                "snippet": _safe_text(snippets[i] if i < len(snippets) else "", max_chars=420),
            }
        )
    return docs


def search_web(
    query: str,
    provider: str,
    api_key: str,
    count: int = 5,
    timeout_seconds: float = 10.0,
) -> list[dict[str, str]]:
    name = str(provider or "brave").strip().lower()
    if name == "brave":
        return brave_web_search(
            query=query,
            api_key=api_key,
            count=count,
            timeout_seconds=timeout_seconds,
        )
    if name in {"wikipedia", "wiki"}:
        return wikipedia_opensearch(
            query=query,
            count=count,
            timeout_seconds=timeout_seconds,
        )
    return []