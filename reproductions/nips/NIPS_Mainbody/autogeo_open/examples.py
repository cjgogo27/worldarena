"""
Example usage of AutoGeo - Autonomous Subagent Scheduling System
"""

import sys

sys.path.insert(0, "/data/alice/cjtest")

from autogeo_open import MasterAgent, initialize_tools, BeliefState, SubAgent, AgentStatus
from autogeo_open.utils import Logger, format_results, MetricsCalculator


def example_single_image():
    """Example: Single image geolocation with single agent"""
    print("\n" + "=" * 60)
    print("Example 1: Single Image Geolocation")
    print("=" * 60)

    # Initialize tools
    initialize_tools()
    logger = Logger("SingleImage")

    # Create master agent
    master = MasterAgent(master_id="master_001")
    logger.info("Created Master Agent")

    # Initialize with single image
    images = ["image_paris_001"]
    master.initialize(images, strategy="default")
    logger.info(f"Initialized with {len(images)} image(s)")

    # Run geolocation
    logger.info("Starting geolocation...")
    results = master.run()

    # Display results
    print(format_results(results))

    return results


def example_multi_image():
    """Example: Multi-image geolocation with multiple agents"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Image Joint Geolocation")
    print("=" * 60)

    # Initialize tools
    initialize_tools()
    logger = Logger("MultiImage")

    # Create master agent
    master = MasterAgent(master_id="master_002", communication_interval=2)
    logger.info("Created Master Agent with staged communication")

    # Initialize with multiple images (simulating MultiLoc dataset)
    images = ["image_paris_beach", "image_paris_mountain", "image_paris_eiffel"]
    master.initialize(images, strategy="default")
    logger.info(f"Initialized with {len(images)} images")

    # Run with more rounds
    master.max_rounds = 15

    # Run geolocation
    logger.info("Starting joint geolocation...")
    results = master.run()

    # Display results
    print(format_results(results))

    # Calculate metrics
    predictions = [
        {"location": p["hypothesis"], "confidence": p["confidence"]}
        for p in results.get("final_predictions", [])
    ]

    recall_1km = MetricsCalculator.recall_at_k(predictions, 1)
    recall_25km = MetricsCalculator.recall_at_k(predictions, 25)
    median_err = MetricsCalculator.median_error(predictions)

    print(f"\nMetrics:")
    print(f"  Recall@1km: {recall_1km:.2%}")
    print(f"  Recall@25km: {recall_25km:.2%}")
    print(f"  Median Error: {median_err:.2f} km")

    return results


def example_grouped_agents():
    """Example: Grouped subagent strategy"""
    print("\n" + "=" * 60)
    print("Example 3: Grouped Subagent Strategy")
    print("=" * 60)

    # Initialize tools
    initialize_tools()

    # Create master agent
    master = MasterAgent(master_id="master_003")

    # Define tool permissions for different agent groups
    tool_permissions = {
        "visual_analyzer": ["image_zoom_tool", "ocr_tool", "image_retrieval_tool"],
        "map_verifier": ["poi_keyword_search", "poi_detail_query", "static_map_query"],
        "knowledge_reasoner": [
            "geoinformation_search",
            "satellite_map_query",
            "poi_input_tips",
        ],
    }

    # Initialize with grouped strategy
    images = [f"image_{i}" for i in range(3)]
    master.create_grouped_subagents(images, tool_permissions)

    print(f"Created {len(master.subagents)} grouped agents")
    for agent_id, agent in master.subagents.items():
        print(f"  - {agent.name}: {agent.available_tools}")

    # Run
    results = master.run()
    print(format_results(results))

    return results


def example_custom_agent():
    """Example: Create custom agent with specific tools"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Agent Configuration")
    print("=" * 60)

    # Initialize tools
    initialize_tools()

    # Create master
    master = MasterAgent(master_id="master_004")

    # Create custom subagent with limited tools
    agent = master.create_subagent(
        name="ocr_specialist",
        tool_permissions=["ocr_tool", "image_zoom_tool"],
        image_id="image_custom",
        confidence_threshold=0.95,
        max_rounds=5,
    )

    print(f"Created custom agent: {agent.name}")
    print(f"Available tools: {agent.available_tools}")
    print(f"Confidence threshold: {agent.confidence_threshold}")

    # Run single agent step by step
    agent.current_belief.hypothesis = "Analyzing..."

    # Execute tools manually
    result1 = agent.call_tool("ocr_tool", image_id="image_custom")
    print(f"\nOCR Result: {result1.success}")

    result2 = agent.call_tool(
        "image_zoom_tool", image_id="image_custom", target_region="sign", zoom_ratio=2.0
    )
    print(f"Zoom Result: {result2.success}")

    # Get summary
    summary = agent.get_execution_summary()
    print(f"\nAgent Summary:")
    print(f"  Total tool calls: {summary['total_tool_calls']}")
    print(f"  Tools used: {summary['tools_used']}")

    return summary


def example_belief_updates():
    """Example: Belief state management"""
    print("\n" + "=" * 60)
    print("Example 5: Belief State Management")
    print("=" * 60)

    # Create belief states
    belief1 = BeliefState(
        hypothesis="Paris, France",
        confidence=0.75,
        image_id="img_001",
        supporting_evidence=["French text detected", "Eiffel Tower visible"],
    )

    belief2 = BeliefState(
        hypothesis="Paris, France",
        confidence=0.85,
        image_id="img_002",
        supporting_evidence=["French architecture", "Champ de Mars nearby"],
    )

    print("Belief 1:")
    print(f"  Hypothesis: {belief1.hypothesis}")
    print(f"  Confidence: {belief1.confidence:.2f}")
    print(f"  Evidence: {belief1.supporting_evidence}")
    print(f"  Level: {belief1.confidence_level}")

    print("\nBelief 2:")
    print(f"  Hypothesis: {belief2.hypothesis}")
    print(f"  Confidence: {belief2.confidence:.2f}")

    # Update confidence
    belief1.update_confidence(0.1)
    print(f"\nAfter update - Belief 1 confidence: {belief1.confidence:.2f}")

    # Serialize
    print(f"\nSerialized: {belief1.to_dict()}")

    return belief1


def run_all_examples():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("# AutoGeo - Autonomous Subagent Scheduling System")
    print("# Examples")
    print("#" * 60)

    try:
        example_single_image()
        example_multi_image()
        example_grouped_agents()
        example_custom_agent()
        example_belief_updates()

        print("\n" + "#" * 60)
        print("# All examples completed successfully!")
        print("#" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
