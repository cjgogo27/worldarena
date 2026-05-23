# Case Studies

Source: extracted from [`neurips_2026.tex`](./neurips_2026.tex)

## Section Text

Figure `fig:case_studies` visualizes representative examples using the model's *actual* reasoning JSON together with a small, readable subset of the retrieved top-k skills. Rather than presenting prompt templates or hand-written explanations, we show how skill-conditioned inference aligns concrete visual evidence with geographically specific prior knowledge. In the figure, each panel includes the input image, key evidence bullets extracted from the model's own reasoning chain, and retrieved skills that most directly support the final prediction.

### Success Cases

Game `KZ2f6LqzJRyChcg8` (Andorra, Round 1) illustrates a high-precision coarse-to-fine success. The model first anchors on a compact Pyrenean mountain settlement, then uses the wall-mounted street sign `Camí de la Llobatera` and Catalan toponymy to narrow the hypothesis to Andorra. Retrieved skills reinforce exactly these cues: Pyrenean valley layout, Andorran/Catalan signage, and mountain-settlement architecture. The final prediction, Ordino, Andorra, is only 2.3 km from the ground truth, showing that retrieved skills can sharpen a visually grounded hypothesis rather than replace it.

Game `2xnQdwiCve2rHWVt` (Thailand, Round 1) shows how skills help accumulate multiple weak but consistent clues. The model identifies apparent Thai script, black-and-white roadside bollards, a broad divided highway, and an east-west corridor structure, then refines the scene to Highway 12 near Khon Kaen. The retrieved skills mirror this progression from country-level Thailand cues to corridor-level road reasoning. The resulting prediction is within 59.9 km of the target. This example is representative of the setting where skill conditioning is most useful: scenes that are not resolved by a single decisive object, but by combining script, road furniture, and infrastructure layout.

### Failure Case

Game `w3MFlsmvpeCTNUcG` (Russia, Round 1) exposes the main failure mode of the method. Skill-conditioned prediction collapsed to Australia (`au`) with a 13,491 km error, while Direct VLM identified Russia correctly. In this case, the flat, dry roadside scene triggered retrieval of skills associated with Australian outback landscapes. Once those regionally incorrect skills were injected, the model over-committed to the wrong continental prior. This failure clarifies an important limitation: skill conditioning helps when retrieved knowledge is geographically aligned with the image, but can actively harm prediction when visually similar scenes pull in the wrong regional skill cluster.

## Images Referenced

- Andorra success case: `/data/alice/cjtest/NIPS/geoskill/data/georc/KZ2f6LqzJRyChcg8/KZ2f6LqzJRyChcg8_1.png`
- Thailand success case: `/data/alice/cjtest/NIPS/geoskill/data/georc/2xnQdwiCve2rHWVt/2xnQdwiCve2rHWVt_1.png`
- Russia failure case: `/data/alice/cjtest/NIPS/geoskill/data/georc/w3MFlsmvpeCTNUcG/w3MFlsmvpeCTNUcG_1.png`
