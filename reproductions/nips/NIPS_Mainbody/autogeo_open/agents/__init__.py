"""
Agents module for AutoGeo
"""

from .subagent import SubAgent, AgentStatus, TerminationReason, ToolCall
from .master_agent import (
    MasterAgent,
    MasterStatus,
    CommunicationTrigger,
    GlobalEvidence,
)

__all__ = [
    "SubAgent",
    "AgentStatus",
    "TerminationReason",
    "ToolCall",
    "MasterAgent",
    "MasterStatus",
    "CommunicationTrigger",
    "GlobalEvidence",
]
