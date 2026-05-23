"""
AutoGeo: Autonomous Subagent Scheduling with Staged Communication
for Efficient Geolocalization

A multi-agent system for geolocation using staged communication
and autonomous tool scheduling.
"""

from .communication.belief_state import BeliefState, CommunicationMessage
from .agents.subagent import SubAgent, AgentStatus, TerminationReason
from .agents.master_agent import MasterAgent, MasterStatus, GlobalEvidence
from .tools import (
    ImageZoomTool,
    OCRTool,
    ImageRetrievalTool,
    POIInputTips,
    POIKeywordSearch,
    POIDetailQuery,
    StaticMapQuery,
    SatelliteMapQuery,
    GeoInformationSearch,
    TOOL_REGISTRY,
    initialize_tools,
)

__version__ = "1.0.0"
__all__ = [
    "BeliefState",
    "CommunicationMessage",
    "SubAgent",
    "AgentStatus",
    "TerminationReason",
    "MasterAgent",
    "MasterStatus",
    "GlobalEvidence",
    "ImageZoomTool",
    "OCRTool",
    "ImageRetrievalTool",
    "POIInputTips",
    "POIKeywordSearch",
    "POIDetailQuery",
    "StaticMapQuery",
    "SatelliteMapQuery",
    "GeoInformationSearch",
    "TOOL_REGISTRY",
    "initialize_tools",
]
