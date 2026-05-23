"""
Master Agent: Orchestrates subagents and manages staged communication
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import uuid
import time

from ..communication.belief_state import BeliefState, CommunicationMessage
from .subagent import SubAgent, AgentStatus, TerminationReason


class MasterStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    COORDINATING = "coordinating"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class CommunicationTrigger(Enum):
    ROUND_BASED = "round_based"  # Every M rounds
    HIGH_CONFIDENCE = "high_confidence"  # Agent reaches high confidence
    LOW_CONFIDENCE = "low_confidence"  # Agent needs help
    CONFLICT_DETECTED = "conflict"  # Agents have conflicting beliefs
    MASTER_INITIATED = "master_initiated"  # Master decides to coordinate


@dataclass
class GlobalEvidence:
    """Shared evidence board accessible by all agents"""

    facts: Dict[str, Any] = field(default_factory=dict)  # objective facts
    agent_beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def add_fact(self, key: str, value: Any):
        """Add an objective fact to the evidence board"""
        if key not in self.facts:
            self.facts[key] = value

    def update_agent_belief(self, agent_id: str, belief: BeliefState):
        """Update an agent's belief"""
        self.agent_beliefs[agent_id] = belief

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between agent beliefs"""
        conflicts = []
        beliefs = list(self.agent_beliefs.values())

        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i + 1 :]:
                if self._are_conflicting(b1, b2):
                    conflicts.append(
                        {
                            "agent1": b1.agent_id,
                            "agent2": b2.agent_id,
                            "belief1": b1.hypothesis,
                            "belief2": b2.hypothesis,
                        }
                    )

        self.conflicts = conflicts
        return conflicts

    def _are_conflicting(self, b1: BeliefState, b2: BeliefState) -> bool:
        """Check if two beliefs are conflicting"""
        # Simple heuristic: different cities/countries at high confidence
        if b1.confidence > 0.7 and b2.confidence > 0.7:
            # Would need more sophisticated location comparison
            if b1.hypothesis != b2.hypothesis:
                return True
        return False

    def get_all_evidence(self) -> Dict[str, Any]:
        """Get all evidence for agent queries"""
        return {
            "facts": self.facts,
            "beliefs": {k: v.to_dict() for k, v in self.agent_beliefs.items()},
            "conflicts": self.conflicts,
        }


@dataclass
class MasterAgent:
    """
    Master Agent for orchestrating subagents with staged communication.
    """

    master_id: str
    name: str = "MasterAgent"
    communication_interval: int = 3  # Communicate every M rounds

    # State
    status: MasterStatus = MasterStatus.INITIALIZING
    subagents: Dict[str, SubAgent] = field(default_factory=dict)
    global_evidence: GlobalEvidence = field(default_factory=GlobalEvidence)
    communication_history: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 20

    # Callbacks
    on_communication: Optional[Callable] = None
    on_aggregation: Optional[Callable] = None
    on_completion: Optional[Callable] = None

    def __post_init__(self):
        if not self.master_id:
            self.master_id = str(uuid.uuid4())[:8]

    def create_subagent(
        self,
        name: str,
        tool_permissions: List[str],
        image_id: Optional[str] = None,
        confidence_threshold: float = 0.9,
        max_rounds: int = 10,
    ) -> SubAgent:
        """Create and register a new subagent"""
        agent_id = str(uuid.uuid4())[:8]

        agent = SubAgent(
            agent_id=agent_id,
            name=name,
            available_tools=tool_permissions,
            image_id=image_id,
            confidence_threshold=confidence_threshold,
            max_rounds=max_rounds,
        )

        # Set up callbacks
        agent.on_belief_update = lambda belief: self._handle_belief_update(
            agent_id, belief
        )
        agent.on_communication_request = lambda: self._handle_communication_request(
            agent_id
        )
        agent.on_termination = lambda reason: self._handle_agent_termination(
            agent_id, reason
        )

        self.subagents[agent_id] = agent
        return agent

    def create_grouped_subagents(
        self, images: List[str], tool_permissions: Dict[str, List[str]] = None
    ) -> Dict[str, SubAgent]:
        """
        Create subagents for multi-image scenario.

        Args:
            images: List of image IDs to analyze
            tool_permissions: Optional dict mapping agent names to their tools
        """
        if tool_permissions is None:
            # Default tool assignment
            tool_permissions = {
                "visual_analyzer": [
                    "image_zoom_tool",
                    "ocr_tool",
                    "image_retrieval_tool",
                ],
                "map_verifier": [
                    "poi_keyword_search",
                    "poi_detail_query",
                    "static_map_query",
                ],
                "knowledge_reasoner": ["geoinformation_search", "satellite_map_query"],
            }

        agents = {}
        agent_names = list(tool_permissions.keys())

        for i, image_id in enumerate(images):
            agent_name = agent_names[i % len(agent_names)]
            agent = self.create_subagent(
                name=f"{agent_name}_{i}",
                tool_permissions=tool_permissions[agent_name],
                image_id=image_id,
            )
            agents[agent.agent_id] = agent

        return agents

    def _handle_belief_update(self, agent_id: str, belief: BeliefState):
        """Handle belief update from a subagent"""
        self.global_evidence.update_agent_belief(agent_id, belief)

    def _handle_communication_request(self, agent_id: str):
        """Handle communication request from a subagent"""
        self._initiate_staged_communication()

    def _handle_agent_termination(self, agent_id: str, reason: TerminationReason):
        """Handle subagent termination"""
        print(f"Agent {agent_id} terminated: {reason.value}")

    def initialize(self, images: List[str], strategy: str = "default"):
        """Initialize the system with images"""
        self.status = MasterStatus.INITIALIZING

        if strategy == "default":
            # Default: create one agent per image with full toolset
            for i, image_id in enumerate(images):
                tool_set = [
                    "image_zoom_tool",
                    "ocr_tool",
                    "image_retrieval_tool",
                    "poi_input_tips",
                    "poi_keyword_search",
                    "poi_detail_query",
                    "static_map_query",
                    "satellite_map_query",
                    "geoinformation_search",
                ]
                self.create_subagent(
                    name=f"agent_{i}", tool_permissions=tool_set, image_id=image_id
                )

        elif strategy == "grouped":
            self.create_grouped_subagents(images)

        self.status = MasterStatus.RUNNING

    def run(self) -> Dict[str, Any]:
        """Run the multi-agent geolocalization"""
        self.status = MasterStatus.RUNNING

        while (
            self.status == MasterStatus.RUNNING and self.current_round < self.max_rounds
        ):
            self.current_round += 1
            print(f"\n=== Round {self.current_round} ===")

            # Execute one step for each subagent
            for agent_id, agent in self.subagents.items():
                if agent.status != AgentStatus.TERMINATED:
                    observations = self._get_observations_for_agent(agent)
                    result = agent.step(observations)

                    print(f"Agent {agent.name}: {result.get('status', 'unknown')}")

                    # Check if communication is needed
                    if result.get("should_communicate", False):
                        self._initiate_staged_communication()

            # Check if all agents are terminated
            if all(a.status == AgentStatus.TERMINATED for a in self.subagents.values()):
                self.status = MasterStatus.AGGREGATING
                break

            # Check for conflicts
            conflicts = self.global_evidence.detect_conflicts()
            if conflicts:
                print(f"Detected {len(conflicts)} conflicts")
                self._resolve_conflicts(conflicts)

        return self._aggregate_results()

    def _get_observations_for_agent(self, agent: SubAgent) -> Dict[str, Any]:
        """Get observations for an agent (could include global evidence)"""
        return {
            "image_id": agent.image_id,
            "global_evidence": self.global_evidence.get_all_evidence(),
            "round": self.current_round,
        }

    def _initiate_staged_communication(self):
        """Initiate staged communication between agents"""
        self.status = MasterStatus.COORDINATING

        print("\n--- Staged Communication ---")

        messages = []
        for agent_id, agent in self.subagents.items():
            if agent.status != AgentStatus.TERMINATED:
                belief = agent.prepare_communication_message()
                message = CommunicationMessage(
                    sender_id=agent_id,
                    receiver_id=None,  # broadcast
                    belief_state=belief,
                    message_type="belief_update",
                )
                messages.append(message)

                # Share belief with other agents
                for other_id, other_agent in self.subagents.items():
                    if (
                        other_id != agent_id
                        and other_agent.status != AgentStatus.TERMINATED
                    ):
                        other_agent.process_incoming_belief(belief)

        # Record communication
        self.communication_history.append(
            {"round": self.current_round, "messages": [m.to_dict() for m in messages]}
        )

        if self.on_communication:
            self.on_communication(messages)

        self.status = MasterStatus.RUNNING

    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]]):
        """Resolve conflicts between agent beliefs"""
        print(f"Resolving {len(conflicts)} conflicts...")

        # Strategy: Request verification from lower confidence agents
        for conflict in conflicts:
            agent1_belief = self.global_evidence.agent_beliefs.get(conflict["agent1"])
            agent2_belief = self.global_evidence.agent_beliefs.get(conflict["agent2"])

            if agent1_belief and agent2_belief:
                # Agent with lower confidence should do more verification
                lower_confidence_agent = (
                    agent1_belief
                    if agent1_belief.confidence < agent2_belief.confidence
                    else agent2_belief
                )

                # Reset agent for more rounds
                agent = self.subagents.get(lower_confidence_agent.agent_id)
                if agent:
                    agent.max_rounds += 3  # Give more time to resolve

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate final results from all agents"""
        self.status = MasterStatus.AGGREGATING

        results = {
            "master_id": self.master_id,
            "total_rounds": self.current_round,
            "agent_results": {},
            "final_predictions": [],
            "confidence_scores": [],
        }

        for agent_id, agent in self.subagents.items():
            summary = agent.get_execution_summary()
            results["agent_results"][agent_id] = summary

            if agent.current_belief:
                results["final_predictions"].append(
                    {
                        "agent_id": agent_id,
                        "image_id": agent.image_id,
                        "hypothesis": agent.current_belief.hypothesis,
                        "confidence": agent.current_belief.confidence,
                        "evidence": agent.current_belief.supporting_evidence,
                    }
                )
                results["confidence_scores"].append(agent.current_belief.confidence)

        # Compute joint prediction
        if results["final_predictions"]:
            avg_confidence = sum(results["confidence_scores"]) / len(
                results["confidence_scores"]
            )
            results["average_confidence"] = avg_confidence
            results["joint_prediction"] = self._compute_joint_prediction(
                results["final_predictions"]
            )

        self.status = MasterStatus.COMPLETED

        if self.on_aggregation:
            self.on_aggregation(results)

        return results

    def _compute_joint_prediction(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute joint prediction from all agent predictions"""
        # Simple strategy: weight by confidence
        if not predictions:
            return {"location": "unknown", "confidence": 0.0}

        # Group by hypothesis
        hypothesis_scores = {}
        for pred in predictions:
            hyp = pred.get("hypothesis", "unknown")
            # Convert to string if it's a dict
            hyp_key = str(hyp) if isinstance(hyp, dict) else hyp
            if hyp_key not in hypothesis_scores:
                hypothesis_scores[hyp_key] = 0.0
            hypothesis_scores[hyp_key] += pred.get("confidence", 0.0)

        # Select highest weighted hypothesis
        best_hypothesis = max(hypothesis_scores.items(), key=lambda x: x[1])

        return {
            "location": best_hypothesis[0],
            "score": best_hypothesis[1] / len(predictions),
            "all_hypotheses": hypothesis_scores,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "master_id": self.master_id,
            "status": self.status.value,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "num_agents": len(self.subagents),
            "active_agents": sum(
                1 for a in self.subagents.values() if a.status != AgentStatus.TERMINATED
            ),
            "communication_count": len(self.communication_history),
            "evidence": self.global_evidence.get_all_evidence(),
        }

    def reset(self):
        """Reset the system"""
        self.subagents = {}
        self.global_evidence = GlobalEvidence()
        self.communication_history = []
        self.current_round = 0
        self.status = MasterStatus.INITIALIZING
