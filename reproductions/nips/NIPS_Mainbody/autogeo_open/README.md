# AutoGeo: Autonomous Subagent Scheduling for Geolocalization

A multi-agent system for multi-image joint geolocalization using staged communication and autonomous tool scheduling.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Master Agent                           │
│  - Orchestrates subagents                                   │
│  - Manages staged communication                             │
│  - Aggregates results                                       │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  SubAgent 1 │     │  SubAgent 2 │     │  SubAgent 3 │
   │  (Visual)   │     │   (Map)     │     │ (Knowledge) │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Tool Pool     │
                    │  - OCR          │
                    │  - Image Zoom   │
                    │  - Map Query    │
                    │  - Geo Search   │
                    └─────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from autogeo import MasterAgent, initialize_tools

# Initialize tools
initialize_tools()

# Create master agent
master = MasterAgent(master_id="geo_001")

# Initialize with images
images = ["img1", "img2", "img3"]
master.initialize(images)

# Run geolocation
results = master.run()

print(results)
```

## Components

### Tools (9 tools)
- **Visual**: `image_zoom_tool`, `ocr_tool`, `image_retrieval_tool`
- **Map**: `poi_input_tips`, `poi_keyword_search`, `poi_detail_query`, `static_map_query`, `satellite_map_query`
- **Knowledge**: `geoinformation_search`

### Agent Types

1. **SubAgent**: Autonomous agent with tool scheduling
   - Makes independent decisions
   - Maintains belief state
   - Can communicate with other agents

2. **MasterAgent**: Orchestrator
   - Creates and manages subagents
   - Handles staged communication
   - Aggregates final predictions

## Examples

See `examples.py` for detailed examples:

1. Single image geolocation
2. Multi-image joint geolocation  
3. Grouped subagent strategy
4. Custom agent configuration
5. Belief state management

## Evaluation Metrics

```python
from autogeo.utils import MetricsCalculator

# Calculate metrics
recall_1km = MetricsCalculator.recall_at_k(predictions, 1)
recall_25km = MetricsCalculator.recall_at_k(predictions, 25)
median_error = MetricsCalculator.median_error(predictions)
geo_score = MetricsCalculator.geo_score(predictions)
consistency = MetricsCalculator.consistency_score(predictions)
```

## Key Features

- **Staged Communication**: Agents communicate every M rounds
- **Autonomous Scheduling**: Each agent decides which tools to call
- **Belief State Management**: Structured belief exchange
- **Conflict Detection**: Identifies contradictory predictions
- **Parallel Execution**: Agents run concurrently

## License

MIT
