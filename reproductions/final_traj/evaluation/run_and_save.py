from run_baselines_and_figures import run_baselines
import pandas as pd
import json

res = run_baselines()
with open('baseline_results.json', 'w') as f:
    json.dump(res, f)
print("Done!")
