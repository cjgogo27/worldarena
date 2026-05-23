from typing import Dict, List, Optional, Any

LANDMARK_TO_CITY = {
    'Eiffel Tower': 'Paris',
    'Louvre': 'Paris',
    'Golden Gate Bridge': 'San Francisco',
    'Statue of Liberty': 'New York',
    'Big Ben': 'London',
    'Colosseum': 'Rome',
    'Great Wall': 'Beijing',
    'Temple of Heaven': 'Beijing',
    'Sydney Opera House': 'Sydney',
    'Taj Mahal': 'Agra'
}

def _city_from_landmarks(features: Dict[str, Any]) -> Optional[str]:
    if not features:
        return None
    landmarks = features.get('landmarks', []) or []
    for m in landmarks:
        if m in LANDMARK_TO_CITY:
            return LANDMARK_TO_CITY[m]
    return None

def _city_from_country(features: Dict[str, Any]) -> Optional[str]:
    if not features:
        return None
    country = features.get('country', '') or ''
    c = country.lower()
    if 'france' in c:
        return 'Paris'
    if 'usa' in c or 'america' in c:
        return 'New York'
    if 'china' in c:
        return 'Beijing'
    if 'japan' in c:
        return 'Tokyo'
    return None

def zoom_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_landmarks(scene.get('features', {}))
    confidence = 0.6 if city else 0.2
    return {"city": city, "confidence": float(confidence), "evidence": ["zoom"]}

def ocr_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    features = scene.get('features', {})
    text = features.get('signText', '') or features.get('text', '') or ''
    city = None
    if 'Rue' in text or 'Street' in text:
        city = 'Paris'
    confidence = 0.25 if city else 0.05
    return {"city": city, "confidence": float(confidence), "evidence": ["ocr", text]} 

def image_retrieval_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_landmarks(scene.get('features', {}))
    return {"city": city, "confidence": float(0.45 if city else 0.15), "evidence": ["retrieval"]}

def poi_keyword_search_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_country(scene.get('features', {}))
    return {"city": city, "confidence": float(0.4 if city else 0.1), "evidence": ["poi_keyword"]}

def poi_detail_query_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_country(scene.get('features', {}))
    return {"city": city, "confidence": float(0.35 if city else 0.08), "evidence": ["poi_detail"]}

def static_map_query_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_country(scene.get('features', {}))
    return {"city": city, "confidence": float(0.3 if city else 0.05), "evidence": ["static_map"]}

def satellite_map_query_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_country(scene.get('features', {}))
    return {"city": city, "confidence": float(0.28 if city else 0.05), "evidence": ["satellite_map"]}

def geoinformation_search_tool(scene: Dict[str, Any]) -> Dict[str, Any]:
    city = _city_from_landmarks(scene.get('features', {})) or _city_from_country(scene.get('features', {}))
    return {"city": city, "confidence": float(0.32 if city else 0.08), "evidence": ["geo_info"]}

__all__ = [
    'zoom_tool', 'ocr_tool', 'image_retrieval_tool', 'poi_keyword_search_tool',
    'poi_detail_query_tool', 'static_map_query_tool', 'satellite_map_query_tool', 'geoinformation_search_tool'
]
