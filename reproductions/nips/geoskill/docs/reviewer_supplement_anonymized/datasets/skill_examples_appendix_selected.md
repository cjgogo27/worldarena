# Representative Skill Examples for Appendix

- Selected from Top30 package: 3 skills (ranks: 1, 7, 11)
- Selection principles: high score, high transferability, and complementary error-control behaviors.

| Rank | Region | Conf | Source Kind | Skill Text |
|---:|---|---:|---|---|
| 1 | asia | 0.95 | fused | Composed skill: if cue 'cyrillic text' co-occurs with cyrillic text, soviet-style roadside design, prioritize asia. Supporting patterns: Composed rule: Cyrillic plus generic ex-Soviet infrastructure narrows to Central Asia, but country choice needs unique local markers such as signage, plates, or mountain context. \| In Central Asia, avoid overcommitting to Kazakhstan on generic rural gas-station scenes without clear national signage. |
| 7 | unknown | 0.97 | recovered | Downweight hemisphere, climate, and generic utility poles when multiple countries share them; make country calls from unique text, traffic side, plates, or sign design. |
| 11 | south_america | 0.94 | recovered | Dry tropical scrub and red dirt are not enough to move from Brazil to East Africa; confirm continent with utility poles, road signage, and vehicle conventions. |

## Canonical Skill Record Format

```json
{
  "skill_text": "<natural-language heuristic>",
  "region_hint": "<europe|asia|north_america|south_america|africa|oceania|unknown>",
  "confidence": <float in [0,1]>,
  "visual_cues": ["cue1", "cue2", "..."],
  "source_game_id": "<origin sample or fusion id>",
  "source_round": <int>
}
```

## Concrete Example (from selected set)

```json
{
  "skill_text": "Composed skill: if cue 'cyrillic text' co-occurs with cyrillic text, soviet-style roadside design, prioritize asia. Supporting patterns: Composed rule: Cyrillic plus generic ex-Soviet infrastructure narrows to Central Asia, but country choice needs unique local markers such as signage, plates, or mountain context. | In Central Asia, avoid overcommitting to Kazakhstan on generic rural gas-station scenes without clear national signage.",
  "region_hint": "asia",
  "confidence": 0.95,
  "visual_cues": [
    "country-specific signage",
    "cyrillic text",
    "generic central asian pavement and poles",
    "license plate style",
    "mountain backdrop",
    "open roadside setting",
    "rural fuel station",
    "soviet-style roadside design"
  ],
  "source_game_id": "skill_fusion",
  "source_round": 0
}
```
