skillgeo sample pack (5 good + 3 bad)

Source predictions:
- /data/alice/cjtest/NIPS/geoskill/experiments/full_100_mytokenland_combined/external_geovista_skill_graph/latest_predictions.json

Selection rule (updated):
- Good: top 5 with smallest geodesic error among samples where retrieved_skill_count > 0.
- Bad: top 3 with largest geodesic error among all valid-coordinate samples.

Package structure:
- selection_index.csv
- cases/<case_name>/input.*
- cases/<case_name>/core_summary.json
- cases/<case_name>/reasoning_and_skill.json
- cases/<case_name>/retrieved_skills.json
- cases/<case_name>/raw_record.json
