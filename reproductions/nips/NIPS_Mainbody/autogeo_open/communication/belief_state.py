"""
Belief State: Structured belief exchange between subagents
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class BeliefState:
    """Represents a subagent's belief about the location"""

    hypothesis: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    uncertainty: str = ""
    requested_validation: List[str] = field(default_factory=list)
    image_id: Optional[str] = None
    agent_id: Optional[str] = None
    round_number: int = 0
    raw_observations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.agent_id is None:
            self.agent_id = str(uuid.uuid4())[:8]

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def add_evidence(self, evidence: str):
        """Add supporting evidence to the belief"""
        if evidence not in self.supporting_evidence:
            self.supporting_evidence.append(evidence)

    def add_validation_request(self, request: str):
        """Request another agent to validate something"""
        if request not in self.requested_validation:
            self.requested_validation.append(request)

    def update_confidence(self, delta: float):
        """Update confidence by a delta value"""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "image_id": self.image_id,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "supporting_evidence": self.supporting_evidence,
            "uncertainty": self.uncertainty,
            "requested_validation": self.requested_validation,
            "round_number": self.round_number,
        }

    def __str__(self) -> str:
        return (
            f"BeliefState(agent={self.agent_id}, image={self.image_id}, "
            f"hypothesis='{self.hypothesis}', confidence={self.confidence:.2f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class CommunicationMessage:
    """Message sent during staged communication"""

    sender_id: str
    receiver_id: Optional[str]  # None = broadcast
    belief_state: BeliefState
    message_type: str = (
        "belief_update"  # belief_update, validation_request, conflict_alert
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "belief": self.belief_state.to_dict(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
