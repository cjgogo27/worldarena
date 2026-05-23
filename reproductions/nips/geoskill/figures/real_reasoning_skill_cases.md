# Real Reasoning + Skill-Matched Casebook

Selected from `experiments/full_100_mytokenland_combined/external_geovista_skill_graph/latest_predictions.json`.
Each case uses the model's real reasoning JSON and aligns evidence bullets to retrieved skills with a lightweight lexical+retrieval-score matcher.

## Case 1: Andorra: Catalan street sign + Pyrenean settlement

- `game_id`: `KZ2f6LqzJRyChcg8`
- `image`: `/data/alice/cjtest/NIPS/geoskill/data/georc/KZ2f6LqzJRyChcg8/KZ2f6LqzJRyChcg8_1.png`
- `prediction`: `Camí de la Llobatera, Ordino, Andorra`
- `distance_km`: `2.261`
- `graph_summary`: `nodes=12, edges=53, base_region=europe, top_skill_region=europe`

**Real Reasoning Summary**

Following the required skill order: (1) the compact built environment in a tight Pyrenean valley fits Andorra well. (2) The signage and vehicle plate format are more consistent with Andorra than nearby Spain or France. (3) This overall evidence strongly places the scene in Andorra. Composing linked supporting skills: (4) the visible street sign style is Andorran, and the road name begins with 'Camí', a Catalan form strongly associated with Andorra; (5) Catalan-language toponymy supports Andorra; (6) the mountainous, forested terrain matches the Pyrenees; (7) the road environment and markings are compatible with Andorra; (10) the stone-and-stucco mountain architecture is characteristic of Pyrenean settlements. While the higher-priority generic city skill points toward Andorra la Vella at country level, the explicit street-name sign visible on the wall reads 'Camí de la Llobatera', which refines the location more precisely to Ordino, Andorra.

**Selected Evidence Bullets**

- Wall-mounted street sign reads 'Camí de la Llobatera'
- Catalan word 'Camí' indicates Catalan-speaking area, strongly fitting Andorra
- Pyrenean terrain with steep forested hills and valley setting
- Stone retaining walls, stucco houses, wooden balconies, steep roofs

**Matched Skills (manually curated from retrieved skills)**

- [rank 1] The compact urban layout fits Andorra la Vella, the capital city, which is set within tight Pyrenean valleys.
  Source: `74bPHM081cMUaNKT` round `3`; retrieval_score: `0.934`
- [rank 2] The signage is not common in Spain and France and the plate is not a typical European license plate with the blue strip on the left, leading to Andorra being the most likely
  Source: `74bPHM081cMUaNKT` round `3`; retrieval_score: `0.842`
- [rank 5] The presence of multiple languages (Catalan, Spanish) on shopfronts further supports Andorra as the location.
  Source: `74bPHM081cMUaNKT` round `3`; retrieval_score: `0.796`
- [rank 6] The terrain is mountainous and forested, indicating a location in or near the Pyrenees.
  Source: `74bPHM081cMUaNKT` round `3`; retrieval_score: `0.714`
- [rank 10] This location is particularly distinct due to the exposed rocky mountain in the background and the abundance of grey brick architecture, commonly found in the countries surrounding the Pyrenees like Northern Spain, Andorra, and Southern France
  Source: `74bPHM081cMUaNKT` round `3`; retrieval_score: `0.651`

**All Raw Evidence Bullets**

- Wall-mounted street sign reads 'Camí de la Llobatera'
- Catalan word 'Camí' indicates Catalan-speaking area, strongly fitting Andorra
- Narrow paved road in a compact mountain settlement
- Pyrenean terrain with steep forested hills and valley setting
- Stone retaining walls, stucco houses, wooden balconies, steep roofs
- Right-hand traffic road layout
- Andorran-style village infrastructure including roadside water channel and guardrail

## Case 2: Kyrgyzstan: gas-station sign + ex-Soviet roadside cues

- `game_id`: `1NJsXTxIF9GGMDxC`
- `image`: `/data/alice/cjtest/NIPS/geoskill/data/georc/1NJsXTxIF9GGMDxC/1NJsXTxIF9GGMDxC_1.png`
- `prediction`: `rural roadside near Osh, Osh Region, Kyrgyzstan`
- `distance_km`: `141.178`
- `graph_summary`: `nodes=12, edges=53, base_region=asia, top_skill_region=asia`

**Real Reasoning Summary**

Applying the ordered skills in sequence: (1) the green branding on the gas station sign is reminiscent of common regional fuel station logos in Kyrgyzstan. (2) With Cyrillic on the gas station sign and fall coverage, this is most likely Kyrgyzstan, though Russia or Ukraine are weaker alternatives. (3) The gas station sign shows Cyrillic script, indicating a country using this alphabet in the former Soviet sphere. (4) The visible word 'АЗС' is the standard Russian abbreviation for a gas station and strongly supports a post-Soviet location. I then compose linked supporting cues from the graph: (6) the fuel prices appear in a format like 45.00, 40.00, 35.00, which fits Kyrgyz fuel pricing better than Kazakhstan, Uzbekistan, or Tajikistan. (8) The deciduous roadside trees and continental seasonal look fit Central Asia and the broader ex-Soviet interior. (11) The white-painted bases of the trees are a common ex-Soviet roadside practice and are especially familiar in Kyrgyzstan. Considering these linked cues together, I reject the contradictory lower-priority alternatives pointing to Thailand, Ukraine, or Kazakhstan because they conflict with the stronger fuel-sign and pricing evidence. Following the final graph conclusion, (12) the best consistent estimate is Osh, Kyrgyzstan, specifically a rural roadside on the outskirts of Osh in Osh Region.

**Selected Evidence Bullets**

- Green roadside fuel price board
- Cyrillic text on the sign
- The word 'АЗС' indicating a gas station
- Fuel prices appearing around 35.00 to 45.00
- White-painted bottoms of roadside trees

**Matched Skills (manually curated from retrieved skills)**

- [rank 1] The green branding on the gas station sign is reminiscent of common regional fuel station logos in Kyrgyzstan.
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.959`
- [rank 2] With the Cyrillic written on the gas station sign and also fall coverage, this is most likely to be Kyrgyzstan, but could possibly be Russia or Ukraine
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.756`
- [rank 4] The word "АЗС" stands for "gas station" in Russian, prevalent in post-Soviet states.
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.666`
- [rank 6] The prices on the sign are in format "45.00", "40.00", "35.00", possibly suggesting Central Asian countries where the ruble or other local currencies use similar formatting.
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.559`
- [rank 11] White painted bottoms of the trees are very common in ex-Soviet countries, but especially common in Kyrgyzstan
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.532`
- [rank 12] Therefore, the best estimate is Osh, Kyrgyzstan.
  Source: `1NJsXTxIF9GGMDxC` round `1`; retrieval_score: `0.528`

**All Raw Evidence Bullets**

- Green roadside fuel price board
- Cyrillic text on the sign
- The word 'АЗС' indicating a gas station
- Fuel prices appearing around 35.00 to 45.00
- White-painted bottoms of roadside trees
- Rural ex-Soviet road and fencing style
- Deciduous trees and dry continental autumn landscape

## Case 3: Thailand: Thai script + divided highway corridor

- `game_id`: `2xnQdwiCve2rHWVt`
- `image`: `/data/alice/cjtest/NIPS/geoskill/data/georc/2xnQdwiCve2rHWVt/2xnQdwiCve2rHWVt_1.png`
- `prediction`: `Highway 12 on the outskirts of Khon Kaen, Khon Kaen Province, Thailand`
- `distance_km`: `59.905`
- `graph_summary`: `nodes=12, edges=64, base_region=asia, top_skill_region=asia`

**Real Reasoning Summary**

Applying the ordered skill graph strictly: Skill 1 gives the primary hypothesis of a rural area near Khon Kaen, Thailand. Skill 2 strengthens this with signage that appears to use Thai script, supporting Thailand over neighboring countries. Via the linked edge from skill 2 to skill 3, the highway orientation and broad divided layout support an east-west corridor, making Highway 12 a plausible fit near Khon Kaen. Via the linked edge from skill 2 to skill 4, the black-and-white roadside bollard style is consistent with Thailand. Skill 5 introduces an alternative of being near a major city such as Bangkok, but this is weaker than the combined Thailand-plus-Khon-Kaen hypothesis from skills 1 through 4. Lower-priority conflicting skills suggesting Vietnam or right-hand traffic are inconsistent with the stronger Thailand cues and the overall road environment, so they are rejected in favor of the higher-priority chain. The most consistent final guess is therefore a peri-urban stretch of Highway 12 on the outskirts of Khon Kaen, Thailand.

**Selected Evidence Bullets**

- Apparent Thai script on roadside signage
- Broad divided multi-lane highway with median island
- Black-and-white roadside bollard style typical of Thailand
- Road appears to run along an east-west corridor consistent with Highway 12
- Peri-urban infrastructure suggests proximity to a major provincial city like Khon Kaen

**Matched Skills (manually curated from retrieved skills)**

- [rank 1] Based on these pieces of evidence, the most likely location is a rural area near Khon Kaen, Thailand.
  Source: `8Uo6ejwXYqmp9av3` round `1`; retrieval_score: `0.874`
- [rank 2] The road signage uses what appears to be Thai script, visible on the center traffic bollard.
  Source: `2xnQdwiCve2rHWVt` round `1`; retrieval_score: `0.861`
- [rank 3] The highway stretches directly east, so it could be some stretch of east-west highway north of Bangkok, or highway 12 (which is mostly east-west)
  Source: `2xnQdwiCve2rHWVt` round `1`; retrieval_score: `0.828`
- [rank 4] The black and white bollard (street post) on the side of the road is also common in Southeast Asian countries but is most common in Thailand
  Source: `5l0GTCFZI877KxkV` round `2`; retrieval_score: `0.818`
- [rank 5] Considering the size of the road and the infrastructure, it suggests proximity to a major city; this is likely near Bangkok, Thailand.
  Source: `2xnQdwiCve2rHWVt` round `1`; retrieval_score: `0.812`
- [rank 12] The cars are driving on the left side of the road, which suggests a country with left-hand traffic such as Thailand, Indonesia, or some countries in Africa.
  Source: `2xnQdwiCve2rHWVt` round `1`; retrieval_score: `0.749`

**All Raw Evidence Bullets**

- Apparent Thai script on roadside signage
- Broad divided multi-lane highway with median island
- Black-and-white roadside bollard style typical of Thailand
- Road appears to run along an east-west corridor consistent with Highway 12
- Peri-urban infrastructure suggests proximity to a major provincial city like Khon Kaen
- Utility poles and overhead wiring match common Thai roadside infrastructure

## Case 4: Japan: narrow concrete farm road + dense utility wiring

- `game_id`: `G3aNW5xo5JUCnAhB`
- `image`: `/data/alice/cjtest/NIPS/geoskill/data/georc/G3aNW5xo5JUCnAhB/G3aNW5xo5JUCnAhB_1.png`
- `prediction`: `Rural lane near farmland in Tsukuba, Ibaraki Prefecture, Japan`
- `distance_km`: `66.945`
- `graph_summary`: `nodes=12, edges=55, base_region=asia, top_skill_region=asia`

**Real Reasoning Summary**

Applying the ordered skill graph strictly: Skill 1 is the strongest starting point, as the narrow concrete road with a sharp edge and shallow drainage canal is highly typical of rural Japan. From Skill 1, the linked supporting skills that remain visually consistent are Skill 2, Skill 3, Skill 4, Skill 6, and Skill 8. Skill 2 supports Japan through the detached rural houses with sloped roofs and Japanese residential proportions. Skill 3 matches the shed-like agricultural outbuildings and simple utilitarian rural construction commonly seen in Japan. Skill 4 is supported by the cultivated plots, orderly agricultural divisions, and peri-urban farming layout. Skill 8 is strongly consistent with the tall utility poles and dense overhead power-line configuration typical in Japan. Skill 6 contributes rural low-density settlement structure and open land, though the flatter plain here fits mainland lowland Japan better than Hokkaido specifically. The alternative linked skills suggesting Thailand, Bangladesh, or India are not consistent with the visible house forms, drainage design, field layout, and pole infrastructure, so they are rejected in the final synthesis. The very flat agricultural landscape with scattered detached homes is especially characteristic of the Kanto plain. Within that setting, Tsukuba in Ibaraki Prefecture is a strong match because it combines broad flat farmland, rural residential lanes, and this exact style of roadside drainage and utility infrastructure.

**Selected Evidence Bullets**

- Very narrow paved rural lane with no centerline markings
- Concrete roadside drainage or irrigation gutter with metal grates
- Japanese-style detached houses with sloped roofs and simple facades
- Agricultural plots directly adjacent to homes in a flat plain
- Utility poles with dense overhead wiring typical of Japan

**Matched Skills (manually curated from retrieved skills)**

- [rank 1] The narrow concrete road with a sharp edge and shallow canal is typical of rural infrastructure in Japan.
  Source: `5l0GTCFZI877KxkV` round `4`; retrieval_score: `0.897`
- [rank 2] The style of the rural houses in the background, with their sloped metal roofs, is characteristic of Japan, especially snowy regions.
  Source: `8G5DpHP2KCtVKrk9` round `4`; retrieval_score: `0.853`
- [rank 3] The architecture of the shed and greenhouse-like structures is commonly found in Japanese rural areas, with the use of corrugated metal and minimalist design.
  Source: `5l0GTCFZI877KxkV` round `4`; retrieval_score: `0.837`
- [rank 4] The vegetation, field type, and clear divisions of agricultural plots are typical of Japanese rural and peri-urban areas.
  Source: `8G5DpHP2KCtVKrk9` round `4`; retrieval_score: `0.813`
- [rank 8] The power lines and their specific configuration are consistent with those found in Japan.
  Source: `9oZfZYQEl9GWjZPu` round `2`; retrieval_score: `0.765`

**All Raw Evidence Bullets**

- Very narrow paved rural lane with no centerline markings
- Concrete roadside drainage or irrigation gutter with metal grates
- Japanese-style detached houses with sloped roofs and simple facades
- Agricultural plots directly adjacent to homes in a flat plain
- Utility poles with dense overhead wiring typical of Japan
- Rural shed and greenhouse-like farm structures
- Low-density settlement pattern in an open cultivated landscape
- Overall flat Kanto-like agricultural terrain consistent with Ibaraki

## Case 5: Montenegro: karst hills + ex-Yugoslav rural house style

- `game_id`: `6ypQOh9cOoE7WaWH`
- `image`: `/data/alice/cjtest/NIPS/geoskill/data/georc/6ypQOh9cOoE7WaWH/6ypQOh9cOoE7WaWH_1.png`
- `prediction`: `Rural local road in the karst hills near Cetinje, Cetinje Municipality, Montenegro`
- `distance_km`: `39.248`
- `graph_summary`: `nodes=12, edges=54, base_region=europe, top_skill_region=europe`

**Real Reasoning Summary**

Following the ordered skill graph strictly: Skill 1 is the strongest starting point: the terrain is clearly hilly with exposed rocky, pale soil and scrubby vegetation, which fits a karst landscape in the Western Balkans. Skill 2 adds that this is a narrow rural residential-type road with a rough light surface and no urban infrastructure, pointing to a minor countryside road in Eastern/Southern Europe rather than a major route. From Skill 1 and Skill 2, the linked cues most consistent through the graph are Skill 5, Skill 6, Skill 10, and Skill 12: the light-colored rough pavement is common in former Yugoslav rural areas; the isolated simple rectangular white house with a red-tiled roof is very typical of rural Croatia/Bosnia and Herzegovina/Montenegro; the lack of lane markings and the road narrowness suggest southern Europe; and the road curvature with surrounding foothill terrain supports a mountainous Adriatic hinterland setting. Skills 3, 4, 7, 9, and 11 are less consistent with the full scene: Bulgaria is weaker than the western Adriatic karst fit, Germany/Czechia do not match the overall vegetation, road width, and house style, southern India/Sri Lanka is incompatible with the terrain and road context, and Scandinavian wooden-house cues are not present. Combining the high-priority terrain signal with the linked former Yugoslav architectural and road-surface cues gives the best consistent match as Montenegro. Within Montenegro, the dry limestone hills and rural setting fit especially well around Cetinje in the inland karst area.

**Selected Evidence Bullets**

- Hilly to mountainous terrain with visible pale rocky karstic ground
- Light-colored rough road surface typical of minor local roads
- Isolated white rectangular house with a red/orange tiled sloped roof
- Overall ex-Yugoslav rural architectural character
- Foothill landscape consistent with the Adriatic inland karst belt

**Matched Skills (manually curated from retrieved skills)**

- [rank 1] The terrain is hilly with rocky soil visible in some places, which fits with the karst landscapes found in the Western Balkans.
  Source: `6ypQOh9cOoE7WaWH` round `1`; retrieval_score: `0.898`
- [rank 5] The road has a rough, light-colored surface, common in rural areas of southern Europe, especially the former Yugoslav countries.
  Source: `6ypQOh9cOoE7WaWH` round `1`; retrieval_score: `0.765`
- [rank 6] The lone house has simple rectangular architecture with a red-tiled roof, a design frequently seen in rural Croatia, Bosnia and Herzegovina, and Montenegro.
  Source: `6ypQOh9cOoE7WaWH` round `1`; retrieval_score: `0.757`
- [rank 10] Absence of lane markings and the narrowness of the road suggest a rural location in southern Europe.
  Source: `9lNwy1vjD53PTSwt` round `2`; retrieval_score: `0.697`
- [rank 12] The curve and landscape of the road suggest a mountainous area, likely in the foothills rather than high altitude, which is common in southern Europe.
  Source: `3uP6lYo9pzx5Q0km` round `5`; retrieval_score: `0.693`

**All Raw Evidence Bullets**

- Hilly to mountainous terrain with visible pale rocky karstic ground
- Very narrow rural paved road without center lines or shoulder
- Light-colored rough road surface typical of minor local roads
- Sparse scrub and mixed deciduous greenery in a dry sunny environment
- Isolated white rectangular house with a red/orange tiled sloped roof
- Overall ex-Yugoslav rural architectural character
- Foothill landscape consistent with the Adriatic inland karst belt

