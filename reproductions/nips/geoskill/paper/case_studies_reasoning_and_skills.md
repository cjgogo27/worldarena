# Case Studies: Skills and Concrete Reasoning Examples

This note splits the new `Case Studies` section into reusable blocks for discussion with Owen.

Important note: the `Matched Retrieved Skills` in this file are raw runtime retrieval snippets, not the richer paper-facing `skill object` schema. For normalized skill objects in the target format, see `case_studies_skill_objects.md`.

## Case 1: Andorra Success

**Paper Summary**

Game `KZ2f6LqzJRyChcg8` is a clean coarse-to-fine success. The model first recognizes a compact Pyrenean mountain settlement, then uses the visible street sign `Camí de la Llobatera` and Catalan toponymy to narrow the hypothesis to Andorra. The final prediction is `Camí de la Llobatera, Ordino, Andorra`, only `2.3 km` from the ground truth.

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/KZ2f6LqzJRyChcg8/KZ2f6LqzJRyChcg8_1.png`

**Concrete Reasoning Example**

- The model anchors on a compact built environment in a tight Pyrenean valley.
- It notes that the signage and plate format fit Andorra better than nearby Spain or France.
- It uses the street sign `Camí de la Llobatera` as a strong local clue.
- It combines Catalan toponymy, mountainous forested terrain, and Pyrenean-style stone-and-stucco settlement architecture.
- It refines the guess from country-level Andorra to `Ordino, Andorra`.

**Matched Retrieved Skills**

- `The compact urban layout fits Andorra la Vella, the capital city, which is set within tight Pyrenean valleys.`
- `The signage is not common in Spain and France and the plate is not a typical European license plate with the blue strip on the left, leading to Andorra being the most likely`
- `The presence of multiple languages (Catalan, Spanish) on shopfronts further supports Andorra as the location.`
- `The terrain is mountainous and forested, indicating a location in or near the Pyrenees.`
- `This location is particularly distinct due to the exposed rocky mountain in the background and the abundance of grey brick architecture, commonly found in the countries surrounding the Pyrenees like Northern Spain, Andorra, and Southern France`

**Concrete Evidence Bullets**

- `Wall-mounted street sign reads 'Camí de la Llobatera'`
- `Catalan word 'Camí' indicates Catalan-speaking area, strongly fitting Andorra`
- `Pyrenean terrain with steep forested hills and valley setting`
- `Stone retaining walls, stucco houses, wooden balconies, steep roofs`

## Case 2: Thailand Success

**Paper Summary**

Game `2xnQdwiCve2rHWVt` shows how skills help accumulate several individually weak but mutually consistent clues. The model combines Thai script, black-and-white roadside bollards, and a broad divided east-west highway corridor, then refines the scene to `Highway 12 on the outskirts of Khon Kaen, Thailand`. The final prediction is within `59.9 km` of the target.

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/2xnQdwiCve2rHWVt/2xnQdwiCve2rHWVt_1.png`

**Concrete Reasoning Example**

- The model starts with a primary hypothesis of rural Thailand near Khon Kaen.
- It strengthens this with apparent Thai script on the roadside signage.
- It uses the broad divided highway and corridor orientation to hypothesize an east-west road, especially Highway 12.
- It adds the black-and-white bollard style as another Thailand-specific cue.
- It rejects weaker alternatives such as Vietnam or right-hand-traffic countries because they conflict with the stronger Thai evidence chain.

**Matched Retrieved Skills**

- `Based on these pieces of evidence, the most likely location is a rural area near Khon Kaen, Thailand.`
- `The road signage uses what appears to be Thai script, visible on the center traffic bollard.`
- `The highway stretches directly east, so it could be some stretch of east-west highway north of Bangkok, or highway 12 (which is mostly east-west)`
- `The black and white bollard (street post) on the side of the road is also common in Southeast Asian countries but is most common in Thailand`
- `Considering the size of the road and the infrastructure, it suggests proximity to a major city; this is likely near Bangkok, Thailand.`

**Concrete Evidence Bullets**

- `Apparent Thai script on roadside signage`
- `Broad divided multi-lane highway with median island`
- `Black-and-white roadside bollard style typical of Thailand`
- `Road appears to run along an east-west corridor consistent with Highway 12`
- `Peri-urban infrastructure suggests proximity to a major provincial city like Khon Kaen`

## Case 3: Failure Case From The Historical Skill-Conditioned Run

This is the failure mode referenced in the current paper draft. It comes from the historical file:

- `/data/alice/cjtest/NIPS/geoskill/experiments/full_100/skill_conditioned/latest_predictions.json`

**Paper Summary**

Game `w3MFlsmvpeCTNUcG` is a useful failure example. In this historical run, the skill-conditioned model collapsed to `Australia` with a very large error, even though the scene is actually in Russia. The main issue is that flat open farmland and generic road cues pulled in a misleading Australia-oriented prior.

**Image**

- `/data/alice/cjtest/NIPS/geoskill/data/georc/w3MFlsmvpeCTNUcG/w3MFlsmvpeCTNUcG_1.png`

**Concrete Reasoning Example**

- The model describes a straight paved road through flat agricultural land with plowed fields on both sides.
- It emphasizes white edge lines and a white center line as being consistent with Australia.
- It interprets the open, sparse, temperate-to-semi-arid agricultural landscape as inland southeastern Australia.
- It then commits to the Murray-Darling Basin style hypothesis instead of keeping Russia active as a serious alternative.

**Retrieved Skills Associated With The Error**

- `The bollards (street posts with reflectors) on the side of the road are unique to Australia, but possibly found in countries in Europe but this climate can never be Europe`
- `The weeds on the side of the road (that have a fan-like shape) are commonly found near and in the state of Terrangnu`
- `The aridness and dirt road coverage leads to the area around Puerto Natales as similar landscapes south of Puerto Natales normally has triple solid road lines on paved roads`
- `The car is following us on the left side of the road and there is a solid yellow single line in the middle, which all Western African countries have`
- `Mail boxes on the side of the road appear to be the American kind`

**Why This Failure Is Useful**

- The top retrieved list is noisy and only weakly grounded in the actual scene.
- A generic road-and-climate prior became stronger than truly discriminative local evidence.
- This case illustrates that retrieval helps only when the injected regional prior is geographically aligned with the image.

## Source Files

- Main paper section: `/data/alice/cjtest/NIPS/geoskill/paper/neurips_2026.tex`
- Success-case casebook: `/data/alice/cjtest/NIPS/geoskill/figures/real_reasoning_skill_cases.md`
- Success-case JSON: `/data/alice/cjtest/NIPS/geoskill/figures/real_reasoning_skill_cases.json`
- Historical failure run: `/data/alice/cjtest/NIPS/geoskill/experiments/full_100/skill_conditioned/latest_predictions.json`
