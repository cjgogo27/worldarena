import json
import pandas as pd
from pathlib import Path

ROOT = Path('/data/alice/cjtest/FinalTraj')

GT_FILE = ROOT / 'California' / 'processed_data_1' / 'all_user_schedules.json'
PERSON_FILE = ROOT / 'California' / 'processed_data_1' / 'california_person_static.json'
HOUSEHOLD_FILE = ROOT / 'California' / 'processed_data_1' / 'california_household_static.json'

BASELINE_FILES = {
    'DeepMove': ROOT / 'Trajectory_Generation_tradition' / 'output_trajectories' / 'deepmove_trajectories_20260325_042142_California.json',
    'LSTPM': ROOT / 'Trajectory_Generation_tradition2' / 'output_trajectories' / 'lstpm_trajectories_20260325_042153_California.json',
    'Indiv-Base': ROOT / 'Trajectory_Generation' / 'output' / 'all_trajectories_20251117_123227.json',
    'Indiv-CoPB': ROOT / 'Trajectory_Generation' / 'output_copb' / 'all_trajectories_20251124_121424.json',
    'HH-Base': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_123218.json',
    'HH-RAG': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_162803.json',
    'HoMe-LLM': ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251117_122412.json',
}

def load_data():
    gt_list = json.load(open(GT_FILE))
    gt = {str(item['user_id']): item for item in gt_list}
    persons = json.load(open(PERSON_FILE))
    hhs = {str(hh['household_id']): hh for hh in json.load(open(HOUSEHOLD_FILE))}
    person_df = pd.DataFrame(persons)
    person_df['household_id'] = person_df['user_id'].apply(lambda x: str(x).split('_')[0])
    person_df['household_income'] = person_df['household_id'].map(lambda h: hhs[h].get('household_income', 'Unknown') if h in hhs else 'Unknown')
    person_df = person_df.set_index('user_id')
    return gt, person_df

gt, person_df = load_data()

models_data = {'Ground Truth': gt}
for name, p in BASELINE_FILES.items():
    if p.exists():
        d_list = json.load(open(p))
        models_data[name] = {str(item.get('user_id', item.get('uid', ''))): item for item in d_list}

def has_act(item, act_name):
    sched = item.get('schedule', item.get('trajectory', []))
    for eps in sched:
        if isinstance(eps, dict):
            act = eps.get('activity', eps.get('activity_type', ''))
            if act and act.lower() == act_name.lower():
                return True
        elif isinstance(eps, str) and eps.lower() == act_name.lower():
            return True
    return False

rows = []
for m_name, data_dict in models_data.items():
    for uid, item in data_dict.items():
        if uid not in person_df.index: continue
        rows.append({
            'Model': m_name,
            'user_id': uid,
            'income': person_df.loc[uid, 'household_income'],
            'gender': person_df.loc[uid, 'gender'],
            'has_exercise': has_act(item, 'exercise'),
            'has_shopping': has_act(item, 'shopping'),
            'has_service': has_act(item, 'service'),
            'has_pickup_dropoff': has_act(item, 'dropoff_pickup') or has_act(item, 'pickup_dropoff') or has_act(item, 'pickup'),
        })
df = pd.DataFrame(rows)

INCOME_ORDER = [
    'Less than $10,000', '$10,000 to $14,999', '$15,000 to $24,999', 
    '$25,000 to $34,999', '$35,000 to $49,999', '$50,000 to $74,999', 
    '$75,000 to $99,999', '$100,000 to $124,999', '$125,000 to $149,999', 
    '$150,000 to $199,999', '$200,000 or more'
]
df['income'] = pd.Categorical(df['income'], categories=INCOME_ORDER, ordered=True)

# Function to safely create a pivot table rounding means to 4 decimals
def create_pivot(df, idx, val):
    # .mean() gives the proportion (since it's True=1, False=0)
    pt = df.groupby([idx, 'Model'])[val].mean().unstack()
    models = ['Ground Truth'] + [k for k in BASELINE_FILES.keys() if k in pt.columns]
    pt = pt[models]
    # Filter rows with NaNs if index is categorical (some tiers might be empty?), although we keep them usually
    return pt.fillna(0.0).round(4)

# Income Table for Exercise & Shopping
inc_ex = create_pivot(df, 'income', 'has_exercise')
inc_sh = create_pivot(df, 'income', 'has_shopping')

inc_ex.to_csv(ROOT / 'evaluation' / 'income_exercise_bias.csv')
inc_sh.to_csv(ROOT / 'evaluation' / 'income_shopping_bias.csv')

# Gender Table for Shopping, Service, and Pickup/Dropoff
gen_sh = create_pivot(df, 'gender', 'has_shopping')
gen_ser = create_pivot(df, 'gender', 'has_service')
gen_pd = create_pivot(df, 'gender', 'has_pickup_dropoff')

gen_sh.to_csv(ROOT / 'evaluation' / 'gender_shopping_bias.csv')
gen_ser.to_csv(ROOT / 'evaluation' / 'gender_service_bias.csv')
gen_pd.to_csv(ROOT / 'evaluation' / 'gender_pickup_dropoff_bias.csv')

# Format as nice Markdown purely for stdout
md_inc_ex = inc_ex.to_markdown()
md_inc_sh = inc_sh.to_markdown()
md_gen_sh = gen_sh.to_markdown()
md_gen_ser = gen_ser.to_markdown()
md_gen_pd = gen_pd.to_markdown()

print("## Income - Exercise")
print(md_inc_ex)
print("\n## Income - Shopping")
print(md_inc_sh)
print("\n## Gender - Shopping")
print(md_gen_sh)
print("\n## Gender - Service")
print(md_gen_ser)
print("\n## Gender - Pickup/Dropoff")
print(md_gen_pd)

