"""
SubAgent: Autonomous agent with tool scheduling capabilities
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import uuid
import time

from ..communication.belief_state import BeliefState
from ..tools import BaseTool, get_tool, TOOL_REGISTRY


class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMMUNICATING = "communicating"
    TERMINATED = "terminated"


class TerminationReason(Enum):
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    TOOLS_EXHAUSTED = "tools_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds"
    EXPLICIT_STOP = "explicit_stop"
    CONFLICT_DETECTED = "conflict_detected"


@dataclass
class ToolCall:
    """Record of a tool call"""

    tool_name: str
    parameters: Dict[str, Any]
    result: Any = None
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result": self.result.to_dict()
            if hasattr(self.result, "to_dict")
            else str(self.result),
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SubAgent:
    """
    Autonomous SubAgent for geolocalization.
    Each agent has its own toolset and makes autonomous decisions.
    """

    agent_id: str
    name: str
    available_tools: List[str]  # List of tool names
    image_id: Optional[str] = None
    current_belief: Optional[BeliefState] = None
    tool_call_history: List[ToolCall] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    round_number: int = 0

    # Configuration
    confidence_threshold: float = 0.9
    max_rounds: int = 10
    min_confidence_improvement: float = 0.05

    # Callbacks
    on_belief_update: Optional[Callable] = None
    on_communication_request: Optional[Callable] = None
    on_termination: Optional[Callable] = None

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())[:8]

        # Initialize belief state
        self.current_belief = BeliefState(
            hypothesis="unknown",
            confidence=0.0,
            image_id=self.image_id,
            agent_id=self.agent_id,
        )

    def get_available_tool_objects(self) -> List[BaseTool]:
        """Get actual tool objects for available tool names"""
        tools = []
        for tool_name in self.available_tools:
            tool = get_tool(tool_name)
            if tool:
                tools.append(tool)
        return tools

    def can_call_tool(self, tool_name: str) -> bool:
        """Check if agent can call a specific tool"""
        return tool_name in self.available_tools

    def call_tool(self, tool_name: str, **kwargs) -> ToolCall:
        """Execute a tool call"""
        if not self.can_call_tool(tool_name):
            return ToolCall(
                tool_name=tool_name,
                parameters=kwargs,
                success=False,
                error=f"Tool {tool_name} not available to agent {self.agent_id}",
            )

        tool = get_tool(tool_name)
        if not tool:
            return ToolCall(
                tool_name=tool_name,
                parameters=kwargs,
                success=False,
                error=f"Tool {tool_name} not found",
            )

        self.status = AgentStatus.EXECUTING
        start_time = time.time()

        try:
            result = tool.execute(**kwargs)
            execution_time = time.time() - start_time

            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=kwargs,
                result=result,
                execution_time=execution_time,
                success=result.success,
            )

            if not result.success:
                tool_call.error = result.error

        except Exception as e:
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=kwargs,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

        self.tool_call_history.append(tool_call)
        self.status = AgentStatus.THINKING

        return tool_call

    def observe(self, image_data: Any) -> Dict[str, Any]:
        """Process input image and extract initial observations"""
        self.status = AgentStatus.THINKING

        observations = {
            "image_id": self.image_id,
            "visual_features": self._extract_features(image_data),
            "initial_impressions": [],
        }

        return observations

    def _extract_features(self, image_data: Any) -> Dict[str, Any]:
        """Extract visual features from image"""
        # In real system, this would use computer vision models
        return {
            "scene_type": "urban",
            "buildings_present": True,
            "signs_present": True,
            "vegetation_visible": True,
            "sky_present": True,
            "time_of_day": "day",
        }

    def decide_next_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what action to take next"""
        if self.round_number >= self.max_rounds:
            return {
                "action": "terminate",
                "reason": TerminationReason.MAX_ROUNDS_REACHED,
            }

        # Simple decision logic - in production would use LLM
        tool_calls_made = len(self.tool_call_history)

        if tool_calls_made == 0:
            # First action: analyze the image with OCR
            return {
                "action": "call_tool",
                "tool_name": "ocr_tool",
                "params": {"image_id": self.image_id},
            }

        elif tool_calls_made == 1:
            # Second action: zoom into details
            return {
                "action": "call_tool",
                "tool_name": "image_zoom_tool",
                "params": {
                    "image_id": self.image_id,
                    "target_region": "sign",
                    "zoom_ratio": 2.5,
                },
            }

        elif tool_calls_made == 2:
            # Third action: search for location
            return {
                "action": "call_tool",
                "tool_name": "geoinformation_search",
                "params": {
                    "query": "Paris landmarks architecture",
                    "knowledge_type": "general",
                },
            }

        elif tool_calls_made == 3:
            # Fourth action: map query
            return {
                "action": "call_tool",
                "tool_name": "poi_keyword_search",
                "params": {"keyword": "landmark", "limit": 5},
            }

        elif tool_calls_made == 4:
            # Fifth action: static map
            return {
                "action": "call_tool",
                "tool_name": "static_map_query",
                "params": {"location": {"lat": 48.8566, "lng": 2.3522}, "zoom": 15},
            }

        elif self._check_confidence_threshold():
            return {
                "action": "terminate",
                "reason": TerminationReason.CONFIDENCE_THRESHOLD,
            }

        else:
            # Terminate after exhausting tools
            return {
                "action": "terminate",
                "reason": TerminationReason.TOOLS_EXHAUSTED,
            }

    def _check_confidence_threshold(self) -> bool:
        """Check if confidence threshold is met"""
        if (
            self.current_belief
            and self.current_belief.confidence >= self.confidence_threshold
        ):
            return True
        return False

    def update_belief(self, tool_call: Any):
        """Update belief state based on tool results"""
        if not self.current_belief:
            return

        # Extract result from ToolCall
        tool_result = tool_call.result if hasattr(tool_call, "result") else tool_call

        if (
            tool_call.success
            and tool_result
            and hasattr(tool_result, "data")
            and tool_result.data
        ):
            # Extract relevant information and update belief
            self.current_belief.update_confidence(0.1)

            # Add evidence from tool result
            data = tool_result.data
            if isinstance(data, dict):
                if "full_text" in data:
                    self.current_belief.add_evidence(
                        f"OCR detected: {data['full_text']}"
                    )
                elif "results" in data:
                    self.current_belief.add_evidence(
                        f"Found {len(data['results'])} relevant results"
                    )
                elif "pois" in data:
                    self.current_belief.add_evidence(f"Found {len(data['pois'])} POIs")

                # Update hypothesis based on results
                if "location" in data:
                    self.current_belief.hypothesis = data["location"]
                elif "text_regions" in data:
                    # Extract text from OCR
                    texts = [r.get("text", "") for r in data.get("text_regions", [])]
                    if texts:
                        self.current_belief.hypothesis = f"Text: {', '.join(texts[:2])}"

        # Notify belief update callback
        if self.on_belief_update:
            self.on_belief_update(self.current_belief)

    def should_communicate(self) -> bool:
        """Determine if agent should initiate communication"""
        # Trigger communication every 3 rounds
        if self.round_number > 0 and self.round_number % 3 == 0:
            return True

        # Trigger on high confidence
        if self.current_belief and self.current_belief.confidence >= 0.85:
            return True

        # Trigger on low confidence (need help)
        if self.current_belief and self.current_belief.confidence < 0.3:
            return True

        return False

    def prepare_communication_message(self) -> BeliefState:
        """Prepare belief state for communication"""
        if self.current_belief:
            self.current_belief.round_number = self.round_number
            return self.current_belief
        return BeliefState(
            hypothesis="unknown",
            confidence=0.0,
            agent_id=self.agent_id,
            image_id=self.image_id,
        )

    def process_incoming_belief(self, belief: BeliefState):
        """Process incoming belief from another agent"""
        # Update own belief based on incoming information
        if self.current_belief and belief:
            # If other agent has higher confidence, consider their hypothesis
            if belief.confidence > self.current_belief.confidence + 0.2:
                # Could update hypothesis here
                pass

    def step(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step of the agent loop"""
        self.round_number += 1
        self.status = AgentStatus.THINKING

        # Decide next action
        decision = self.decide_next_action(observations)

        if decision["action"] == "terminate":
            self.status = AgentStatus.TERMINATED
            if self.on_termination:
                self.on_termination(decision.get("reason"))
            return {"status": "terminated", "reason": decision.get("reason")}

        if decision["action"] == "call_tool":
            tool_name = decision["tool_name"]
            params = decision.get("params", {})

            result = self.call_tool(tool_name, **params)
            self.update_belief(result)

            return {
                "status": "tool_executed",
                "tool": tool_name,
                "result": result.to_dict() if hasattr(result, "to_dict") else result,
                "should_communicate": self.should_communicate(),
            }

        return {"status": "waiting"}

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of agent execution"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "image_id": self.image_id,
            "total_rounds": self.round_number,
            "total_tool_calls": len(self.tool_call_history),
            "final_belief": self.current_belief.to_dict()
            if self.current_belief
            else None,
            "status": self.status.value,
            "tools_used": list(set(tc.tool_name for tc in self.tool_call_history)),
        }

    def reset(self):
        """Reset agent state for reuse"""
        self.current_belief = BeliefState(
            hypothesis="unknown",
            confidence=0.0,
            image_id=self.image_id,
            agent_id=self.agent_id,
        )
        self.tool_call_history = []
        self.round_number = 0
        self.status = AgentStatus.IDLE
