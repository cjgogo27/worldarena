# Case Studies: Normalized Skill Objects

This file corrects the earlier mismatch between:

- the current repo runtime skill schema, and
- the richer paper-facing `skill object` schema you want to present.

## What Is Real vs. What Is Normalized

### Current Runtime Skill In The Repo

The current runtime skill stored by the codebase is the minimal object defined in `src/skill_parser.py`:

```python
@dataclass
class Skill:
    skill_text: str
    region_hint: str
    confidence: float
    visual_cues: list[str]
    source_game_id: str
    source_round: int
```

### Paper-Facing Normalized Skill Object

Below, each skill is rewritten into the target object format. To stay honest:

- `confidence` comes from the real runtime skill.
- `source_chain_refs` are traced back to the original source chain files.
- `skill_id`, `name`, `level`, `trigger_condition`, `observation_targets`, `evidence_type`, `reasoning_action`, `applicable_regions`, and `risk_flags` are normalized for presentation from the raw runtime skill snippet.

## Case 1: Andorra Success

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/KZ2f6LqzJRyChcg8/KZ2f6LqzJRyChcg8_1.png`

**Concrete Reasoning Example**

- The model identifies a compact Pyrenean mountain settlement.
- It uses the sign `Camí de la Llobatera` and Catalan toponymy to move from a border-region hypothesis to Andorra.
- It further uses stone-and-stucco mountain architecture and steep forested valley terrain to refine the guess to Ordino.

### Skill Object A1

```json
{
  "skill_id": "sk_catalan_signage_andorra_v1",
  "name": "Use Catalan-looking roadside or storefront text to narrow to Andorra",
  "level": "L1-L2",
  "trigger_condition": "image contains readable street-sign or storefront text in a compact Pyrenean settlement",
  "observation_targets": ["road_sign_text", "storefront_text", "toponymy"],
  "evidence_type": "language",
  "reasoning_action": "narrow_down_region",
  "applicable_regions": ["Andorra", "Catalan-speaking Pyrenees"],
  "risk_flags": ["ocr_error_possible", "tourist_area_may_be_multilingual"],
  "confidence": 0.75,
  "source_chain_refs": ["74bPHM081cMUaNKT:expert_r3:s2"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The signage is not common in Spain and France and the plate is not a typical European license plate with the blue strip on the left, leading to Andorra being the most likely`

### Skill Object A2

```json
{
  "skill_id": "sk_pyrenean_valley_layout_andorra_v1",
  "name": "Use compact Pyrenean valley settlement layout to support an Andorra hypothesis",
  "level": "L1-L2",
  "trigger_condition": "image shows a compact mountain town embedded in a steep forested valley",
  "observation_targets": ["settlement_layout", "terrain_profile", "vegetation_type"],
  "evidence_type": "terrain",
  "reasoning_action": "support_country_hypothesis",
  "applicable_regions": ["Andorra", "Pyrenees"],
  "risk_flags": ["mountain_regions_can_look_similar", "cross_border_overlap_in_pyrenees"],
  "confidence": 0.65,
  "source_chain_refs": ["74bPHM081cMUaNKT:candidate_gpt4_r3:s7"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The compact urban layout fits Andorra la Vella, the capital city, which is set within tight Pyrenean valleys.`

### Skill Object A3

```json
{
  "skill_id": "sk_pyrenees_stone_architecture_v1",
  "name": "Use Pyrenean stone-and-brick mountain architecture to disambiguate Andorra-like scenes",
  "level": "L1-L2",
  "trigger_condition": "image shows grey stone or brick buildings in a mountainous Southern European town",
  "observation_targets": ["architecture_style", "building_material", "terrain_profile"],
  "evidence_type": "architecture",
  "reasoning_action": "narrow_down_region",
  "applicable_regions": ["Andorra", "Northern Spain", "Southern France"],
  "risk_flags": ["shared_architecture_across_border_regions", "renovation_style_can_blur_signal"],
  "confidence": 0.65,
  "source_chain_refs": ["74bPHM081cMUaNKT:expert_r3:s1"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `This location is particularly distinct due to the exposed rocky mountain in the background and the abundance of grey brick architecture, commonly found in the countries surrounding the Pyrenees like Northern Spain, Andorra, and Southern France`

## Case 2: Thailand Success

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/2xnQdwiCve2rHWVt/2xnQdwiCve2rHWVt_1.png`

**Concrete Reasoning Example**

- The model first anchors on Thailand from left-hand traffic and roadside script.
- It then uses the broad divided highway and corridor direction to hypothesize an east-west trunk road.
- It combines bollard style and peri-urban scale to refine toward the Highway 12 / Khon Kaen corridor.

### Skill Object T1

```json
{
  "skill_id": "sk_thai_script_roadsign_v1",
  "name": "Use Thai script on roadside elements to identify Thailand",
  "level": "L1-L2",
  "trigger_condition": "image contains readable script on a traffic bollard, sign, or median marker",
  "observation_targets": ["road_sign_text", "script_type"],
  "evidence_type": "language",
  "reasoning_action": "narrow_down_region",
  "applicable_regions": ["Thailand"],
  "risk_flags": ["ocr_error_possible", "small_text_may_be_low_resolution"],
  "confidence": 0.65,
  "source_chain_refs": ["2xnQdwiCve2rHWVt:candidate_gpt4_r1:s2"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The road signage uses what appears to be Thai script, visible on the center traffic bollard.`

### Skill Object T2

```json
{
  "skill_id": "sk_thailand_bollard_blackwhite_v1",
  "name": "Use black-and-white roadside bollards to support Thailand over nearby alternatives",
  "level": "L1-L2",
  "trigger_condition": "image shows roadside bollards or posts with black-and-white banding",
  "observation_targets": ["bollard_style", "roadside_post_design"],
  "evidence_type": "infrastructure",
  "reasoning_action": "narrow_down_region",
  "applicable_regions": ["Thailand", "Mainland Southeast Asia"],
  "risk_flags": ["shared_road_furniture_across_neighbors", "partial_visibility_may_hide_pattern"],
  "confidence": 0.65,
  "source_chain_refs": ["5l0GTCFZI877KxkV:expert_r2:s2"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The black and white bollard (street post) on the side of the road is also common in Southeast Asian countries but is most common in Thailand`

### Skill Object T3

```json
{
  "skill_id": "sk_highway12_khonkaen_corridor_v1",
  "name": "Use broad east-west divided-highway cues to refine toward the Khon Kaen and Highway 12 corridor",
  "level": "L2-L3",
  "trigger_condition": "image shows a broad divided peri-urban highway with an east-west corridor feel in Thailand",
  "observation_targets": ["road_layout", "highway_orientation", "urban_proximity"],
  "evidence_type": "infrastructure",
  "reasoning_action": "refine_location",
  "applicable_regions": ["Khon Kaen", "Highway 12 corridor", "Thailand"],
  "risk_flags": ["multiple_highways_can_share_similar_design", "view_direction_may_distort_orientation"],
  "confidence": 0.75,
  "source_chain_refs": ["8Uo6ejwXYqmp9av3:candidate_gpt4_r1:s8"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `Based on these pieces of evidence, the most likely location is a rural area near Khon Kaen, Thailand.`

## Case 3: Failure Case With Misaligned Retrieval

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/w3MFlsmvpeCTNUcG/w3MFlsmvpeCTNUcG_1.png`

**Concrete Reasoning Example**

- The wrong branch over-weights generic flat-road and dry-landscape cues.
- Retrieval injects Australia-, Malaysia-, and US-oriented snippets that are only weakly grounded in the actual image.
- The failure is therefore useful because it shows the danger of noisy regional priors.

### Skill Object F1

```json
{
  "skill_id": "sk_australia_reflector_bollard_v1",
  "name": "Use Australian-style roadside reflector bollards to jump to inland Australia",
  "level": "L1-L2",
  "trigger_condition": "image shows roadside reflector posts in an open dry landscape",
  "observation_targets": ["bollard_style", "roadside_post_design", "climate_context"],
  "evidence_type": "infrastructure",
  "reasoning_action": "narrow_down_region",
  "applicable_regions": ["Australia", "Oceania"],
  "risk_flags": ["bollard_patterns_can_be_misread", "similar_open_landscapes_exist_on_other_continents"],
  "confidence": 0.55,
  "source_chain_refs": ["56Q4T4rpv9O9sCpP:expert_r4:s1"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The bollards (street posts with reflectors) on the side of the road are unique to Australia, but possibly found in countries in Europe but this climate can never be Europe`

### Skill Object F2

```json
{
  "skill_id": "sk_terengganu_fan_weeds_v1",
  "name": "Use fan-shaped roadside weeds to localize toward Terengganu in Malaysia",
  "level": "L2-L3",
  "trigger_condition": "image shows fan-shaped roadside weeds in a humid left-driving tropical scene",
  "observation_targets": ["vegetation_type", "roadside_plants"],
  "evidence_type": "vegetation",
  "reasoning_action": "refine_location",
  "applicable_regions": ["Terengganu", "Malaysia"],
  "risk_flags": ["plant_morphology_is_hard_to_verify", "similar_vegetation_exists_across_tropical_regions"],
  "confidence": 0.65,
  "source_chain_refs": ["1NJsXTxIF9GGMDxC:expert_r2:s5"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `The weeds on the side of the road (that have a fan-like shape) are commonly found near and in the state of Terrangnu`

### Skill Object F3

```json
{
  "skill_id": "sk_american_mailbox_suburban_us_v1",
  "name": "Use American-style roadside mailboxes to support a US suburban prior",
  "level": "L1-L2",
  "trigger_condition": "image shows standalone roadside mailboxes in low-density residential context",
  "observation_targets": ["mailbox_style", "housing_layout"],
  "evidence_type": "infrastructure",
  "reasoning_action": "support_country_hypothesis",
  "applicable_regions": ["United States"],
  "risk_flags": ["small_objects_may_be_hallucinated", "similar_mailboxes_can_exist_in_parts_of_Canada"],
  "confidence": 0.65,
  "source_chain_refs": ["AVKTblAzBqaYrcKe:expert_r4:s2"],
  "version": "v1"
}
```

Raw retrieved snippet:

- `Mail boxes on the side of the road appear to be the American kind`
