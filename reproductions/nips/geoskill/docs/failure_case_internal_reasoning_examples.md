# Internal Failure Reasoning Examples (For Figure/Table)

Selection rule: external_error=None and prediction exists, but internal reasoning leads to country mismatch.

## georc::JnTw9kl2nWPFaoUg::round1

- image_path: /data/alice/cjtest/NIPS/geoskill/data/georc/JnTw9kl2nWPFaoUg/JnTw9kl2nWPFaoUg_1.png
- external_error: None
- gt_country: om
- pred_country: ae
- gt_coordinates: (24.856925881889026, 56.40432516086915)
- predicted_address: Al Ain, Abu Dhabi Emirate, United Arab Emirates
- predicted_coordinates: (24.19167, 55.76056)
- geodesic_error_km: 98.6
- confidence: 0.72

### Ground-Truth vs Prediction Contrast

| Field | Ground Truth | Model Prediction |
|---|---|---|
| Country | om (Oman) | ae (United Arab Emirates) |
| Coordinates | (24.856925881889026, 56.40432516086915) | (24.19167, 55.76056) |
| Address Anchor | N/A (dataset coordinate GT) | Al Ain, Abu Dhabi Emirate, United Arab Emirates |
| Distance Error | 0.0 km (reference) | 98.6 km |

### Erroneous Reasoning Excerpt

STEP 1 - HEMISPHERE & CLIMATE: Arid desert environment with sparse drought-tolerant vegetation, consistent with Arabian Peninsula climate. STEP 2 - REGIONAL INDICATORS: White SUV (Toyota Land Cruiser type) is extremely common in UAE/GCC countries. Low concrete block wall with smooth stucco finish and flat roof architecture matches UAE residential/commercial construction style. Unpaved gravel/dirt surface suggests developing area or desert periphery. STEP 3 - LOCALIZATION: Al Ain is specifically known for its oasis agriculture, palm tree landscaping, and mixed developed/undeveloped areas at desert edge. The combination of maintained palm trees with otherwise barren surroundings fits Al Ain's unique character as a 'garden city' in the desert. The hazy atmospheric conditions also match Al Ain's inland desert location. STEP 4 - CONFIDENCE ADJUSTMENT: While UAE is strongly indicated, specific city identification to Al Ain carries moderate uncertainty due to similar characteristics across UAE emirates; however, the palm cultivation context and oasis-adjacent development pattern favors Al Ain over Dubai or Abu Dhabi city proper.

### Evidence Claimed by the Model

- arid desert landscape with minimal natural vegetation
- white SUV roof - Toyota Land Cruiser type common in GCC
- low concrete block wall with smooth stucco finish
- flat roof structure
- palm tree landscaping in otherwise barren area
- unpaved gravel/dirt road surface
- hazy atmospheric conditions typical of inland desert
- young planted trees suggesting managed landscaping/oasis agriculture

### Skill-Trajectory Evidence (What the System Actually Used)

- Stage-1 prior already commits to `ae` (`predicted_country=ae`, `predicted_region=asia`, `confidence=0.65`).
- `active_region=asia`, and 10 skills are retrieved under this broad region gate.
- High-impact retrieved skill: score 0.783, desert + vehicle pattern -> **Mongolia** (strongly off-target).
- High-impact retrieved skill: score 0.656, semi-arid vegetation -> **Jordan/Amman**.
- High-impact retrieved skill: score 0.629, sign style -> **Indonesia**.
- High-impact retrieved skill: score 0.696, haze + missing signs -> **India**.
- Local analysis itself is ambiguous (`implied_country: "United Arab Emirates or Saudi Arabia"`) and does not provide a decisive contradiction.
- Net effect: the synthesis stage is fed by region-level and partially noisy skills, then confirms the initial UAE prior.

### Why This Is an Internal Failure (Deep Root-Cause Analysis)

- Causal step 1: the early stage already commits to `ae` from coarse arid/Gulf signals, so downstream reasoning starts from a strong UAE prior.
- Causal step 2: retrieved skills are mostly region-level (`asia`) and include several low-discriminative or noisy patterns (for example generic desert/GCC cues, plus unrelated country hints), which increases ambiguity instead of separating Oman from UAE.
- Causal step 3: local-detail analysis lacks hard identifiers (for example no decisive sign text, plate format, or unique road markers), so the correction path has weak evidence to overturn the initial hypothesis.
- Causal step 4: final synthesis behaves as confirmation, not falsification, and anchors to a plausible UAE city (`Al Ain`) rather than testing Oman-specific alternatives.
- Root-cause class: low-discriminability cue dominance + retrieval noise + missing counterfactual elimination.

## georc::1NJsXTxIF9GGMDxC::round1

- image_path: /data/alice/cjtest/NIPS/geoskill/data/georc/1NJsXTxIF9GGMDxC/1NJsXTxIF9GGMDxC_1.png
- external_error: None
- gt_country: kg
- pred_country: kz
- gt_coordinates: (41.40303624959777, 74.02089741449993)
- predicted_address: Almaty, Kazakhstan
- predicted_coordinates: (43.25249, 76.9115)
- geodesic_error_km: 314.2
- confidence: 0.75

### Ground-Truth vs Prediction Contrast

| Field | Ground Truth | Model Prediction |
|---|---|---|
| Country | kg (Kyrgyzstan) | kz (Kazakhstan) |
| Coordinates | (41.40303624959777, 74.02089741449993) | (43.25249, 76.9115) |
| Address Anchor | N/A (dataset coordinate GT) | Almaty, Kazakhstan |
| Distance Error | 0.0 km (reference) | 314.2 km |

### Erroneous Reasoning Excerpt

STEP 1 - HEMISPHERE & CLIMATE ZONE: Sun position and shadows suggest Northern Hemisphere, mid-latitude continental climate with dry conditions. STEP 2 - SCRIPT & LANGUAGE: Cyrillic text on the fuel station sign strongly points to a post-Soviet Central Asian country. Kazakhstan is the largest and most populous of these with this landscape. STEP 3 - INFRASTRUCTURE & VEGETATION: Wooden utility poles with crossarms, simple asphalt road with minimal markings, and sparse deciduous trees match rural or semi-urban areas near Almaty in the foothills of the Tian Shan mountains. STEP 4 - COMMERCIAL SIGNAGE: The tall vertical green fuel price sign with circular logo at top resembles the branding of KazMunayGas (Qazaq Oil) or similar Kazakh fuel retailers. STEP 5 - VEHICLES & ROAD FEATURES: The silver/white van and white concrete bollards are consistent with Kazakh road infrastructure. STEP 6 - SYNTHESIS: Combining Cyrillic script, Central Asian arid-temperate landscape, Kazakh fuel station branding, and Soviet-era utility infrastructure, the most probable location is in the Almaty Region, with Almaty city as the best geocoding anchor.

### Evidence Claimed by the Model

- Cyrillic script on green fuel station sign
- wooden utility pole with crossarm and multiple insulators
- dry arid landscape with sparse deciduous trees
- simple asphalt road with no center line and white edge bollards
- tall vertical fuel price sign with circular logo
- silver/white van typical of Central Asian vehicle fleet
- continental climate lighting and sky conditions

### Skill-Trajectory Evidence (What the System Actually Used)

- Stage-1 prior commits to `kz` (`predicted_country=kz`, `predicted_region=asia`, `confidence=0.75`).
- Local analysis is explicitly uncertain (`implied_country: "Kazakhstan or Kyrgyzstan"`), so this stage does not break the tie.
- Retrieved skill (supportive): score 0.689, "Cyrillic gas station sign -> most likely Kyrgyzstan".
- Retrieved skill (supportive): score 0.752, "green fuel branding resembles Kyrgyzstan".
- Retrieved skill (supportive): score 0.709, "greenery suggests southern Kyrgyzstan".
- Retrieved skill (competing noise): score 0.774, desert + vehicle pattern -> **Mongolia**.
- Retrieved skill (competing noise): score 0.836, dry-lighting prior -> **India**.
- Retrieved skill (competing noise): score 0.649, sign-style prior -> **Indonesia**.
- Net effect: although Kyrgyz evidence exists, the system does not explicitly perform contradiction-based elimination and keeps the earlier Kazakhstan anchor.

### Why This Is an Internal Failure (Deep Root-Cause Analysis)

- Causal step 1: stage-1 prediction already commits to `kz`, so subsequent reasoning is conditioned by a Kazakhstan-first hypothesis.
- Causal step 2: retrieved skills contain mixed signals, including one Kyrgyzstan-supporting cue, but several broad Central-Asia/post-Soviet cues remain high-ranked and dominate the synthesis.
- Causal step 3: the cited evidence (Cyrillic text, dry climate, Soviet-style infrastructure, generic fuel-station patterns) is region-discriminative but not country-discriminative within Kyrgyzstan vs Kazakhstan.
- Causal step 4: the model escalates from weak country evidence to a strong city anchor (`Almaty`), which amplifies an early misclassification into a confident final answer.
- Root-cause class: hypothesis lock-in + broad-prior over-weighting + weak contradiction handling in the final fusion step.

## Cross-Case Mechanism: Why These Internal Reasoning Errors Happen

Both examples share the same core mechanism: **early coarse hypothesis lock-in under under-constrained evidence**.

- Shared factor A (evidence granularity mismatch): available visual cues are strong for region-level inference but weak for country-level disambiguation.
- Shared factor B (retrieval specificity gap): region-gated retrieval returns semantically related but country-weak (and sometimes off-target) skills, which contaminate the evidence pool.
- Shared factor C (insufficient negative testing): the pipeline prefers confirming a plausible hypothesis over explicitly eliminating close alternatives.
- Shared factor D (anchor amplification): once a country prior is formed, textual geocoding anchors (major city/province names) make the final output look coherent and high-confidence, even when the initial country choice is wrong.

In short, these are not API/runtime failures; they are **internal inference-structure failures** caused by ambiguous cues, imperfect retrieval specificity, and confirmation-biased synthesis.

