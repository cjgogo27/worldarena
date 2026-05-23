"""
Tool definitions for AutoGeo geolocalization system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import json
import uuid


class ToolCategory(Enum):
    VISUAL = "visual"
    MAP = "map"
    KNOWLEDGE = "knowledge"
    CONTROL = "control"


@dataclass
class ToolResult:
    """Result from a tool execution"""

    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
        }


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Dict[str, Any],
        required_permissions: List[str] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters
        self.required_permissions = required_permissions or []
        self.execution_count = 0

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters - override in subclass"""
        raise NotImplementedError("Subclasses must implement execute method")

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution"""
        required = self.parameters.get("required", [])
        for req_param in required:
            if req_param not in params:
                return False
        return True

    def get_schema(self) -> Dict[str, Any]:
        """Return tool schema for documentation"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
        }


class ImageZoomTool(BaseTool):
    """Zoom into specific regions of an image"""

    def __init__(self):
        super().__init__(
            name="image_zoom_tool",
            description="Zoom into specific regions of an image to see details",
            category=ToolCategory.VISUAL,
            parameters={
                "required": ["image_id", "target_region"],
                "optional": ["zoom_ratio", "keep_context"],
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "ID of the image to zoom",
                    },
                    "target_region": {
                        "type": "string",
                        "description": "Region to zoom (e.g., 'building_facade', 'sign', 'license_plate')",
                    },
                    "zoom_ratio": {
                        "type": "float",
                        "default": 2.0,
                        "description": "Zoom magnification",
                    },
                    "keep_context": {
                        "type": "boolean",
                        "default": True,
                        "description": "Keep surrounding context",
                    },
                },
            },
        )

    def execute(
        self,
        image_id: str,
        target_region: str,
        zoom_ratio: float = 2.0,
        keep_context: bool = True,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        # Simulated execution - in real system would call actual image processing
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "zoomed_image_id": f"{image_id}_zoom_{self.execution_count}",
                "target_region": target_region,
                "zoom_ratio": zoom_ratio,
                "extracted_features": self._simulate_feature_extraction(target_region),
            },
            metadata={"region": target_region, "ratio": zoom_ratio},
        )

    def _simulate_feature_extraction(self, region: str) -> Dict[str, Any]:
        """Simulate feature extraction based on region"""
        feature_map = {
            "building_facade": {"type": "architecture", "style": "modern/traditional"},
            "sign": {"type": "text", "language": "unknown"},
            "license_plate": {"type": "vehicle_id", "format": "unknown"},
            "vegetation": {"type": "flora", "species": "unknown"},
            "street_view": {"type": "urban", "elements": []},
        }
        return feature_map.get(region, {"type": "unknown"})


class OCRTool(BaseTool):
    """Extract text from images using OCR"""

    def __init__(self):
        super().__init__(
            name="ocr_tool",
            description="Extract text from images (signs, license plates,店铺名)",
            category=ToolCategory.VISUAL,
            parameters={
                "required": ["image_id"],
                "optional": ["language_hint", "min_confidence"],
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "ID of the image to process",
                    },
                    "language_hint": {
                        "type": "string",
                        "default": "auto",
                        "description": "Language hint for OCR",
                    },
                    "min_confidence": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Minimum confidence threshold",
                    },
                },
            },
        )

    def execute(
        self,
        image_id: str,
        language_hint: str = "auto",
        min_confidence: float = 0.5,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        # Simulated OCR - would use actual OCR service in production
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "text_regions": self._simulate_ocr(image_id),
                "full_text": " ".join(
                    [r["text"] for r in self._simulate_ocr(image_id)]
                ),
                "detected_languages": ["en", "fr"],  # Simulated
            },
            metadata={"language_hint": language_hint},
        )

    def _simulate_ocr(self, image_id: str) -> List[Dict[str, Any]]:
        """Simulate OCR text extraction"""
        # In real system, this would call actual OCR
        return [
            {"text": "RUE DE LA PAIX", "confidence": 0.95, "region": "sign_top"},
            {"text": "PARIS", "confidence": 0.92, "region": "sign_bottom"},
        ]


class ImageRetrievalTool(BaseTool):
    """Fine-grained image retrieval based on visual features"""

    def __init__(self):
        super().__init__(
            name="image_retrieval_tool",
            description="Search for similar images using visual features",
            category=ToolCategory.VISUAL,
            parameters={
                "required": ["query_image_id"],
                "optional": ["top_k", "database", "filters"],
                "properties": {
                    "query_image_id": {
                        "type": "string",
                        "description": "Image to search for",
                    },
                    "top_k": {
                        "type": "int",
                        "default": 5,
                        "description": "Number of results",
                    },
                    "database": {
                        "type": "string",
                        "default": "street_view",
                        "description": "Image database to search",
                    },
                    "filters": {
                        "type": "object",
                        "default": {},
                        "description": "Optional filters",
                    },
                },
            },
        )

    def execute(
        self,
        query_image_id: str,
        top_k: int = 5,
        database: str = "street_view",
        filters: Dict = None,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "results": self._simulate_retrieval(query_image_id, top_k),
                "query_image": query_image_id,
                "database": database,
            },
            metadata={"top_k": top_k},
        )

    def _simulate_retrieval(self, query_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Simulate image retrieval results"""
        return [
            {
                "image_id": f"sv_{i}",
                "similarity": 0.9 - i * 0.1,
                "location": f"Paris, France",
            }
            for i in range(top_k)
        ]


class POIInputTips(BaseTool):
    """POI search with autocomplete suggestions"""

    def __init__(self):
        super().__init__(
            name="poi_input_tips",
            description="Get POI search suggestions based on partial input",
            category=ToolCategory.MAP,
            parameters={
                "required": ["keyword"],
                "optional": ["location_hint", "max_results"],
                "properties": {
                    "keyword": {"type": "string", "description": "Partial keyword"},
                    "location_hint": {
                        "type": "string",
                        "description": "Location context",
                    },
                    "max_results": {"type": "int", "default": 10},
                },
            },
        )

    def execute(
        self, keyword: str, location_hint: str = None, max_results: int = 10, **kwargs
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "suggestions": self._simulate_suggestions(
                    keyword, location_hint, max_results
                )
            },
            metadata={"keyword": keyword},
        )

    def _simulate_suggestions(
        self, keyword: str, location_hint: str, max_results: int
    ) -> List[Dict[str, Any]]:
        base_results = [
            {"name": f"{keyword} Street", "type": "street"},
            {"name": f"{keyword} Avenue", "type": "avenue"},
            {"name": f"{keyword} Restaurant", "type": "restaurant"},
        ]
        return base_results[:max_results]


class POIKeywordSearch(BaseTool):
    """POI keyword search"""

    def __init__(self):
        super().__init__(
            name="poi_keyword_search",
            description="Search for POIs by keyword",
            category=ToolCategory.MAP,
            parameters={
                "required": ["keyword"],
                "optional": ["location", "radius", "category", "limit"],
                "properties": {
                    "keyword": {"type": "string", "description": "Search keyword"},
                    "location": {
                        "type": "object",
                        "description": "Center location {lat, lng}",
                    },
                    "radius": {
                        "type": "int",
                        "default": 5000,
                        "description": "Search radius in meters",
                    },
                    "category": {"type": "string", "description": "POI category"},
                    "limit": {"type": "int", "default": 20},
                },
            },
        )

    def execute(
        self,
        keyword: str,
        location: Dict = None,
        radius: int = 5000,
        category: str = None,
        limit: int = 20,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "pois": self._simulate_poi_search(keyword, location, limit),
                "total_found": limit,
            },
            metadata={"keyword": keyword, "radius": radius},
        )

    def _simulate_poi_search(
        self, keyword: str, location: Dict, limit: int
    ) -> List[Dict[str, Any]]:
        return [
            {
                "name": f"{keyword} - Location {i}",
                "lat": 48.85 + i * 0.01,
                "lng": 2.35 + i * 0.01,
                "category": "landmark",
            }
            for i in range(min(limit, 5))
        ]


class POIDetailQuery(BaseTool):
    """Get detailed information about a specific POI"""

    def __init__(self):
        super().__init__(
            name="poi_detail_query",
            description="Get detailed information about a specific POI",
            category=ToolCategory.MAP,
            parameters={
                "required": ["poi_id"],
                "optional": ["include_reviews", "include_photos"],
                "properties": {
                    "poi_id": {"type": "string", "description": "POI identifier"},
                    "include_reviews": {"type": "boolean", "default": False},
                    "include_photos": {"type": "boolean", "default": False},
                },
            },
        )

    def execute(
        self,
        poi_id: str,
        include_reviews: bool = False,
        include_photos: bool = False,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "poi_id": poi_id,
                "name": f"POI {poi_id}",
                "address": "123 Example Street, Paris",
                "coordinates": {"lat": 48.8566, "lng": 2.3522},
                "type": "landmark",
                "opening_hours": "09:00-18:00",
            },
            metadata={"requested_fields": ["reviews"] if include_reviews else []},
        )


class StaticMapQuery(BaseTool):
    """Query static map images"""

    def __init__(self):
        super().__init__(
            name="static_map_query",
            description="Get static map images for a location",
            category=ToolCategory.MAP,
            parameters={
                "required": ["location"],
                "optional": ["zoom", "size", "markers"],
                "properties": {
                    "location": {
                        "type": "object",
                        "description": "Center location {lat, lng}",
                    },
                    "zoom": {
                        "type": "int",
                        "default": 15,
                        "description": "Zoom level (1-20)",
                    },
                    "size": {
                        "type": "string",
                        "default": "600x400",
                        "description": "Image size",
                    },
                    "markers": {"type": "array", "description": "Markers to display"},
                },
            },
        )

    def execute(
        self,
        location: Dict,
        zoom: int = 15,
        size: str = "600x400",
        markers: List = None,
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "map_url": f"https://staticmap.example.com?lat={location['lat']}&lng={location['lng']}&zoom={zoom}",
                "location": location,
                "zoom": zoom,
                "size": size,
                "visible_pois": self._simulate_visible_pois(location, zoom),
            },
            metadata={"location": location},
        )

    def _simulate_visible_pois(self, location: Dict, zoom: int) -> List[Dict[str, Any]]:
        return [
            {"name": "Nearby Restaurant", "distance": 100},
            {"name": "Metro Station", "distance": 200},
        ]


class SatelliteMapQuery(BaseTool):
    """Query satellite/aerial imagery"""

    def __init__(self):
        super().__init__(
            name="satellite_map_query",
            description="Get satellite imagery for a location",
            category=ToolCategory.MAP,
            parameters={
                "required": ["location"],
                "optional": ["zoom", "cloud_free"],
                "properties": {
                    "location": {
                        "type": "object",
                        "description": "Center location {lat, lng}",
                    },
                    "zoom": {"type": "int", "default": 17, "description": "Zoom level"},
                    "cloud_free": {"type": "boolean", "default": True},
                },
            },
        )

    def execute(
        self, location: Dict, zoom: int = 17, cloud_free: bool = True, **kwargs
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "satellite_url": f"https://satellite.example.com?lat={location['lat']}&lng={location['lng']}",
                "location": location,
                "zoom": zoom,
                "terrain_features": self._simulate_terrain(location),
            },
            metadata={"location": location, "cloud_free": cloud_free},
        )

    def _simulate_terrain(self, location: Dict) -> Dict[str, Any]:
        return {
            "land_use": "urban",
            "vegetation_density": "moderate",
            "water_bodies": False,
            "terrain_type": "flat",
        }


class GeoInformationSearch(BaseTool):
    """Search geographic information and knowledge"""

    def __init__(self):
        super().__init__(
            name="geoinformation_search",
            description="Search geographic knowledge (climate, culture, history)",
            category=ToolCategory.KNOWLEDGE,
            parameters={
                "required": ["query"],
                "optional": ["location_hint", "knowledge_type"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "location_hint": {
                        "type": "string",
                        "description": "Location context",
                    },
                    "knowledge_type": {
                        "type": "string",
                        "default": "general",
                        "description": "Type: general, climate, culture, history",
                    },
                },
            },
        )

    def execute(
        self,
        query: str,
        location_hint: str = None,
        knowledge_type: str = "general",
        **kwargs,
    ) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "results": self._simulate_geo_search(query, location_hint),
                "query": query,
                "knowledge_type": knowledge_type,
            },
            metadata={"query": query},
        )

    def _simulate_geo_search(
        self, query: str, location_hint: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "title": f"Information about {query}",
                "content": f"Relevant geographic information for {query}...",
                "relevance": 0.9,
                "source": "knowledge_base",
            }
        ]


# Global tool registry
TOOL_REGISTRY: Dict[str, BaseTool] = {}


def initialize_tools() -> Dict[str, BaseTool]:
    """Initialize all available tools"""
    tools = [
        ImageZoomTool(),
        OCRTool(),
        ImageRetrievalTool(),
        POIInputTips(),
        POIKeywordSearch(),
        POIDetailQuery(),
        StaticMapQuery(),
        SatelliteMapQuery(),
        GeoInformationSearch(),
    ]

    global TOOL_REGISTRY
    TOOL_REGISTRY = {tool.name: tool for tool in tools}
    return TOOL_REGISTRY


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Get a tool by name from registry"""
    return TOOL_REGISTRY.get(tool_name)


def get_tools_by_category(category: ToolCategory) -> List[BaseTool]:
    """Get all tools in a specific category"""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_tools_for_permissions(permissions: List[str]) -> List[BaseTool]:
    """Get tools available for given permissions"""
    available = []
    for tool in TOOL_REGISTRY.values():
        if not tool.required_permissions or any(
            p in tool.required_permissions for p in permissions
        ):
            available.append(tool)
    return available
