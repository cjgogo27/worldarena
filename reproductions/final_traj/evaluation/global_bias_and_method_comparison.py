import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import run_baselines_and_figures as rbf


ROOT = Path('/data/alice/cjtest/FinalTraj')
OUT_DIR = ROOT / 'review' / 'Human_Mobility_Generation' / 'fig'

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

COLLAB_ACTS = {'dropoff_pickup', 'socialize', 'dine_out', 'shopping'}
STATE_ORDER = ['Home', 'Work', 'Education', 'ShopServ', 'DropPickup', 'Recreation', 'Other']
METHOD_ORDER = [
    'DeepMove', 'LSTPM', 'MarkovChain', 'Empirical Sampling', 'Rule-based (CDAP)',
    'Indiv-Base', 'Indiv-CoPB', 'HH-Base', 'HH-RAG', 'HoMe-LLM'
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_lookup(rows):
    out = {}
    for r in rows:
        uid = str(r.get('user_id', '')).strip()
        if uid:
            out[uid] = r
    return out


def sequence_to_schedule(seq):
    seq = list(seq)
    if not seq:
        return []
    schedule = []
    start = 0
    prev = int(seq[0])
    for i in range(1, len(seq)):
        cur = int(seq[i])
        if cur != prev:
            schedule.append({
                'start_time': _slot_to_hhmm(start),
                'end_time': _slot_to_hhmm(i),
                'activity': rbf.CODE_TO_ACTIVITY.get(prev, 'home'),
            })
            start = i
            prev = cur
    schedule.append({
        'start_time': _slot_to_hhmm(start),
        'end_time': '24:00',
        'activity': rbf.CODE_TO_ACTIVITY.get(prev, 'home'),
    })
    return schedule


def _slot_to_hhmm(slot_idx):
    minutes = int(slot_idx) * rbf.TIMESTEP_MINUTES
    hh = minutes // 60
    mm = minutes % 60
    if hh >= 24:
        return '24:00'
    return f'{hh:02d}:{mm:02d}'


def to_seq(row):
    if isinstance(row, dict) and 'schedule' in row:
        return rbf.schedule_to_96_timesteps(row['schedule'])
    if isinstance(row, (list, np.ndarray)):
        arr = np.asarray(row, dtype=int)
        if arr.ndim == 1 and arr.shape[0] == rbf.N_TIMESTEPS:
            return arr
    raise ValueError(f'Unsupported row format: {type(row)}')


def collaboration_recall(gt_seq, pred_seq):
    collab_codes = [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS]
    mask = np.isin(gt_seq, collab_codes)
    if mask.sum() == 0:
        return np.nan
    return float(np.logical_and(mask, gt_seq == pred_seq).sum() / mask.sum())


def activity_group(activity_name):
    if activity_name == 'home':
        return 'Home'
    if activity_name == 'work':
        return 'Work'
    if activity_name == 'education':
        return 'Education'
    if activity_name in {'shopping', 'service'}:
        return 'ShopServ'
    if activity_name == 'dropoff_pickup':
        return 'DropPickup'
    if activity_name in {'socialize', 'exercise', 'dine_out'}:
        return 'Recreation'
    return 'Other'


def signature(seq):
    acts = []
    prev = None
    for x in seq:
        act = rbf.CODE_TO_ACTIVITY.get(int(x), 'home')
        if act == 'home':
            continue
        if act != prev:
            acts.append(act)
            prev = act
    return ' > '.join(acts[:6]) if acts else 'home_only'


def fit_synthetic_methods(gt_lookup, person_lookup, hh_lookup, user_ids):
    gt_seqs = np.array([to_seq(gt_lookup[uid]) for uid in user_ids], dtype=int)
    n_train = int(len(gt_seqs) * 0.8)
    train_seqs = gt_seqs[:max(1, n_train)]

    mc = rbf.MarkovChainBaseline()
    mc.fit(train_seqs)
    freq = rbf.FrequencyBaseline()
    freq.fit(train_seqs)
    rule = rbf.RuleBasedHHBaseline(random_state=42)
    rule.fit(train_seqs, person_lookup=person_lookup, hh_lookup=hh_lookup)

    np.random.seed(42)
    mc_gen = mc.generate(len(user_ids))
    np.random.seed(42)
    freq_gen = freq.generate(len(user_ids))
    rule_rows = rule.generate_for_users(user_ids)

    return {
        'MarkovChain': [{'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(user_ids, mc_gen)],
        'Empirical Sampling': [{'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(user_ids, freq_gen)],
        'Rule-based (CDAP)': rule_rows,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style('whitegrid')

    gt_rows = load_json(GT_FILE)
    person_rows = load_json(PERSON_FILE)
    hh_rows = load_json(HOUSEHOLD_FILE)

    gt_lookup = build_lookup(gt_rows)
    person_lookup = build_lookup(person_rows)
    hh_lookup = {str(h.get('household_id', '')): h for h in hh_rows}

    loaded = {}
    sets = [set(gt_lookup.keys())]
    for method, path in BASELINE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing file for {method}: {path}')
        lookup = build_lookup(load_json(path))
        loaded[method] = lookup
        sets.append(set(lookup.keys()))

    common_users = sorted(set.intersection(*sets))
    if not common_users:
        raise RuntimeError('No common users across GT and loaded methods.')

    synthetic = fit_synthetic_methods(gt_lookup, person_lookup, hh_lookup, common_users)

    method_rows = {}
    for method in BASELINE_FILES.keys():
        method_rows[method] = [loaded[method][uid] for uid in common_users]
    for method, rows in synthetic.items():
        method_rows[method] = rows

    # Build user-level table for all methods.
    user_metrics = []
    gt_seqs = {uid: to_seq(gt_lookup[uid]) for uid in common_users}

    # composition rarity defined within evaluated cohort
    comp_counter = Counter()
    user_comp = {}
    for uid in common_users:
        hid = uid.split('_')[0]
        h = hh_lookup.get(hid, {})
        comp = f"S{int(h.get('household_size', 0) or 0)}-A{int(h.get('adult_count', 0) or 0)}-C{int(h.get('young_children_count', 0) or 0)}-V{int(h.get('vehicle_count', 0) or 0)}"
        user_comp[uid] = comp
        comp_counter[comp] += 1

    for method, rows in method_rows.items():
        for uid, row in zip(common_users, rows):
            pred_seq = to_seq(row)
            gt_seq = gt_seqs[uid]
            person = person_lookup.get(uid, {})
            hid = uid.split('_')[0]
            hh = hh_lookup.get(hid, {})

            user_metrics.append({
                'method': method,
                'user_id': uid,
                'household_id': hid,
                'gender': person.get('gender', 'Unknown'),
                'race': person.get('race', 'Unknown'),
                'relationship': person.get('relationship', 'Unknown'),
                'household_income': hh.get('household_income', 'Unknown'),
                'household_race': hh.get('household_race', 'Unknown'),
                'household_size': int(hh.get('household_size', 0) or 0),
                'young_children_count': int(hh.get('young_children_count', 0) or 0),
                'composition': user_comp[uid],
                'composition_freq': comp_counter[user_comp[uid]],
                'is_rare_composition': comp_counter[user_comp[uid]] <= 4,
                'accuracy': float((pred_seq == gt_seq).mean()),
                'collab_recall': collaboration_recall(gt_seq, pred_seq),
                'gt_signature': signature(gt_seq),
                'pred_signature': signature(pred_seq),
            })

    udf = pd.DataFrame(user_metrics)
    udf['is_female'] = udf['gender'].astype(str).str.lower() == 'female'
    udf['is_low_income'] = udf['household_income'].astype(str).str.contains('\$15,000|\$25,000|\$35,000|\$50,000|\$75,000', regex=True)
    udf['is_minority_race'] = ~udf['race'].astype(str).str.lower().eq('white')

    # Method-level bias summary.
    summary_rows = []
    for method in METHOD_ORDER:
        mdf = udf[udf['method'] == method]
        if mdf.empty:
            continue

        female_collab = mdf.loc[mdf['is_female'], 'collab_recall'].mean()
        low_income_acc = mdf.loc[mdf['is_low_income'], 'accuracy'].mean()
        rare_acc = mdf.loc[mdf['is_rare_composition'], 'accuracy'].mean()
        common_acc = mdf.loc[~mdf['is_rare_composition'], 'accuracy'].mean()
        white_acc = mdf.loc[~mdf['is_minority_race'], 'accuracy'].mean()
        minority_acc = mdf.loc[mdf['is_minority_race'], 'accuracy'].mean()

        sig_count = mdf['pred_signature'].value_counts(normalize=True)
        top1_share = float(sig_count.iloc[0]) if len(sig_count) > 0 else np.nan

        summary_rows.append({
            'method': method,
            'overall_accuracy': mdf['accuracy'].mean(),
            'female_collab_recall': female_collab,
            'low_income_accuracy': low_income_acc,
            'rare_comp_accuracy': rare_acc,
            'common_comp_accuracy': common_acc,
            'race_gap_white_minus_minority': white_acc - minority_acc,
            'fixed_pattern_top1_share': top1_share,
        })

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT_DIR / 'global_bias_summary_by_method.csv', index=False)
    udf.to_csv(OUT_DIR / 'global_bias_user_level_metrics.csv', index=False)

    # Figure A: bias overview across methods.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ordered = sdf.set_index('method').loc[[m for m in METHOD_ORDER if m in set(sdf['method'])]].reset_index()

    axes[0, 0].bar(ordered['method'], ordered['female_collab_recall'], color='#E76F51')
    axes[0, 0].set_title('Female Collaboration Recall')
    axes[0, 0].tick_params(axis='x', rotation=35)
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].bar(ordered['method'], ordered['low_income_accuracy'], color='#2A9D8F')
    axes[0, 1].set_title('Low-income Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=35)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(ordered['method'], ordered['race_gap_white_minus_minority'], color='#457B9D')
    axes[1, 0].set_title('Race Gap (White - Minority)')
    axes[1, 0].tick_params(axis='x', rotation=35)
    axes[1, 0].axhline(0, color='black', linewidth=0.8)

    axes[1, 1].bar(ordered['method'], ordered['fixed_pattern_top1_share'], color='#8D99AE')
    axes[1, 1].set_title('Fixed-Pattern Index (Top-1 Signature Share)')
    axes[1, 1].tick_params(axis='x', rotation=35)
    axes[1, 1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'global_bias_method_overview.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'global_bias_method_overview.png', dpi=300)
    plt.close(fig)

    # Figure B: temporal stacked facets (similar style to reference).
    gt_user_df = udf[udf['method'] == 'HoMe-LLM'][['user_id', 'gt_signature']].drop_duplicates()
    top_patterns = gt_user_df['gt_signature'].value_counts().head(4).index.tolist()

    compare_methods = ['GT', 'HoMe-LLM', 'HH-RAG', 'Indiv-Base', 'Rule-based (CDAP)']
    color_map = {
        'Home': '#E9D8A6',
        'Work': '#7F5539',
        'Education': '#7B2CBF',
        'ShopServ': '#D62828',
        'DropPickup': '#F4A261',
        'Recreation': '#3A86FF',
        'Other': '#8D99AE',
    }

    def method_seq(method, uid):
        if method == 'GT':
            return gt_seqs[uid]
        row = method_rows[method][common_users.index(uid)]
        return to_seq(row)

    n_rows = len(top_patterns)
    n_cols = len(compare_methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.5 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.array([axes])

    x = np.arange(rbf.N_TIMESTEPS)
    for i, pat in enumerate(top_patterns):
        pat_users = gt_user_df[gt_user_df['gt_signature'] == pat]['user_id'].tolist()
        for j, method in enumerate(compare_methods):
            ax = axes[i, j]
            if not pat_users:
                ax.axis('off')
                continue

            mat = np.zeros((len(STATE_ORDER), rbf.N_TIMESTEPS), dtype=float)
            for uid in pat_users:
                seq = method_seq(method, uid)
                for t in range(rbf.N_TIMESTEPS):
                    act = rbf.CODE_TO_ACTIVITY.get(int(seq[t]), 'home')
                    grp = activity_group(act)
                    k = STATE_ORDER.index(grp)
                    mat[k, t] += 1.0
            mat = mat / max(1, len(pat_users))

            cum = np.zeros(rbf.N_TIMESTEPS)
            for state in STATE_ORDER:
                k = STATE_ORDER.index(state)
                y = mat[k]
                ax.fill_between(x, cum, cum + y, color=color_map[state], alpha=0.95, linewidth=0)
                cum += y

            if i == 0:
                ax.set_title(method, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'Pattern {i+1}\n(n={len(pat_users)})')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, rbf.N_TIMESTEPS - 1)
            ax.set_xticks([0, 24, 48, 72, 95])
            ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:45'], rotation=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in STATE_ORDER]
    fig.legend(handles, STATE_ORDER, loc='center right', frameon=False)
    fig.suptitle('Temporal Activity Pattern Facets Across Methods', fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    fig.savefig(OUT_DIR / 'method_temporal_pattern_facets.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'method_temporal_pattern_facets.png', dpi=300)
    plt.close(fig)

    # Text summary for paper.
    worst_female = udf[(udf['method'] == 'HoMe-LLM') & (udf['is_female'])].sort_values(['collab_recall', 'accuracy']).head(8)
    report_lines = [
        '# Global Bias Pattern Report (All LLM-cohort users)',
        '',
        f'- Cohort size (common users): {len(common_users)}',
        f'- Methods compared: {len(sdf)}',
        '',
        '## Key patterns observed',
    ]

    if not sdf.empty:
        best_overall = sdf.sort_values('overall_accuracy', ascending=False).iloc[0]
        worst_fixed = sdf.sort_values('fixed_pattern_top1_share', ascending=False).iloc[0]
        report_lines.extend([
            f"- Best overall accuracy method: {best_overall['method']} ({best_overall['overall_accuracy']:.3f})",
            f"- Strongest fixed-pattern bias (top1 signature share): {worst_fixed['method']} ({worst_fixed['fixed_pattern_top1_share']:.3f})",
        ])

    report_lines.extend([
        '',
        '### Female collaboration-failure examples (HoMe-LLM)',
    ])

    for _, r in worst_female.iterrows():
        report_lines.append(
            f"- {r['user_id']}: accuracy={r['accuracy']:.3f}, collab_recall={r['collab_recall']:.3f}, relationship={r['relationship']}, income={r['household_income']}"
        )

    report_lines.extend([
        '',
        '## Generated files',
        '- global_bias_summary_by_method.csv',
        '- global_bias_user_level_metrics.csv',
        '- global_bias_method_overview.pdf/.png',
        '- method_temporal_pattern_facets.pdf/.png',
    ])

    with open(OUT_DIR / 'global_bias_pattern_report.md', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f'Saved: {OUT_DIR / "global_bias_summary_by_method.csv"}')
    print(f'Saved: {OUT_DIR / "global_bias_user_level_metrics.csv"}')
    print(f'Saved: {OUT_DIR / "global_bias_method_overview.pdf"}')
    print(f'Saved: {OUT_DIR / "method_temporal_pattern_facets.pdf"}')
    print(f'Saved: {OUT_DIR / "global_bias_pattern_report.md"}')


if __name__ == '__main__':
    main()
