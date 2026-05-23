import json
import os
from collections import defaultdict
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

GT_FILE = ROOT / 'California' / 'processed_data' / 'all_user_schedules.json'
PERSON_FILE = ROOT / 'California' / 'processed_data' / 'california_person_static.json'
HOUSEHOLD_FILE = ROOT / 'California' / 'processed_data' / 'california_household_static.json'

BASELINE_FILES = {
    'DeepMove': ROOT / 'Trajectory_Generation_tradition' / 'output_trajectories' / 'deepmove_trajectories_20260325_042142_California.json',
    'LSTPM': ROOT / 'Trajectory_Generation_tradition2' / 'output_trajectories' / 'lstpm_trajectories_20260325_042153_California.json',
    'Indiv-Base': ROOT / 'Trajectory_Generation' / 'output' / 'all_trajectories_20251117_123227.json',
    'Indiv-CoPB': ROOT / 'Trajectory_Generation' / 'output_copb' / 'all_trajectories_20251124_121424.json',
    'HH-Base': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_123218.json',
    'HH-RAG': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_162803.json',
    'HoMe-LLM (Ours)': ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251117_122412.json',
}


METRICS = [
    'accuracy', 'edit_dist', 'bleu_score', 'micro_hour', 'macro_int', 'data_jsd', 'act_type', 'traj_len'
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_lookup(records):
    out = {}
    for row in records:
        uid = str(row.get('user_id', '')).strip()
        if uid:
            out[uid] = row
    return out


def user_richness(schedule):
    non_home = [seg for seg in schedule if seg.get('activity') != 'home']
    uniq_non_home = len({seg.get('activity', '') for seg in non_home})
    return uniq_non_home * 100 + len(non_home)


def pick_five_users(gt_lookup, person_lookup, hh_lookup, all_required_user_sets):
    common_users = set(gt_lookup.keys())
    for uset in all_required_user_sets:
        common_users &= uset

    candidates = []
    for uid in common_users:
        person = person_lookup.get(uid)
        if not person:
            continue
        hhid = str(person.get('household_id', person.get('SAMPNO', person.get('sampno', '')))).strip()
        if not hhid:
            # Most person records use user_id like "<household_id>_<person_idx>".
            hhid = uid.split('_')[0]
        hh = hh_lookup.get(hhid)
        if not hh:
            continue
        hh_size = int(hh.get('household_size', 0) or 0)
        if hh_size <= 0:
            continue

        schedule = gt_lookup[uid].get('schedule', [])
        if len(schedule) < 4:
            continue

        completeness_flags = [
            person.get('gender') is not None,
            person.get('age_range') is not None,
            person.get('employment_status') is not None,
            bool(hhid),
            hh.get('household_income') is not None,
        ]
        completeness = sum(1 for x in completeness_flags if x)
        richness = user_richness(schedule)
        score = completeness * 1000 + richness

        candidates.append({
            'user_id': uid,
            'household_id': hhid,
            'household_size': hh_size,
            'score': score,
            'richness': richness,
            'completeness': completeness,
        })

    if not candidates:
        raise RuntimeError('No valid users found for all baselines intersection.')

    by_size = defaultdict(list)
    for c in candidates:
        by_size[c['household_size']].append(c)
    for size in by_size:
        by_size[size].sort(key=lambda x: x['score'], reverse=True)

    selected = []
    for size in sorted(by_size.keys()):
        selected.append(by_size[size][0])
        if len(selected) == 5:
            break

    if len(selected) < 5:
        used = {x['user_id'] for x in selected}
        remaining = sorted(candidates, key=lambda x: x['score'], reverse=True)
        for c in remaining:
            if c['user_id'] in used:
                continue
            selected.append(c)
            used.add(c['user_id'])
            if len(selected) == 5:
                break

    selected = sorted(selected, key=lambda x: (x['household_size'], -x['score']))[:5]
    return selected


def to_seq_array(schedule_rows):
    out = []
    for row in schedule_rows:
        if isinstance(row, dict) and 'schedule' in row:
            out.append(rbf.schedule_to_96_timesteps(row['schedule']))
            continue
        if isinstance(row, dict) and 'sequence' in row:
            out.append(np.asarray(row['sequence'], dtype=int))
            continue
        if isinstance(row, (list, np.ndarray)):
            arr = np.asarray(row, dtype=int)
            if arr.ndim == 1 and arr.shape[0] == rbf.N_TIMESTEPS:
                out.append(arr)
                continue
        raise ValueError(f'Unsupported prediction row format: {type(row)}')
    return np.array(out, dtype=int)


def eval_pack(pred_rows, gt_rows):
    pred = to_seq_array(pred_rows)
    gt = to_seq_array(gt_rows)
    return rbf.evaluate(pred, gt)


def per_user_accuracy(pred_rows, gt_rows):
    values = []
    for p, g in zip(pred_rows, gt_rows):
        if isinstance(p, dict) and 'schedule' in p:
            ps = rbf.schedule_to_96_timesteps(p['schedule'])
        else:
            ps = np.asarray(p, dtype=int)
        gs = rbf.schedule_to_96_timesteps(g['schedule'])
        values.append(float((ps == gs).sum() / len(ps)))
    return values


def _slot_to_hhmm(slot_idx):
    minutes = int(slot_idx) * rbf.TIMESTEP_MINUTES
    hh = minutes // 60
    mm = minutes % 60
    if hh >= 24:
        return '24:00'
    return f'{hh:02d}:{mm:02d}'


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


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rbf.set_publication_style()

    gt_data = load_json(GT_FILE)
    person_data = load_json(PERSON_FILE)
    hh_data = load_json(HOUSEHOLD_FILE)

    gt_lookup = build_lookup(gt_data)
    person_lookup = build_lookup(person_data)
    hh_lookup = {str(x.get('household_id', x.get('SAMPNO', x.get('sampno', '')))): x for x in hh_data}

    baseline_rows = {}
    baseline_user_sets = []
    for method, path in BASELINE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing baseline file for {method}: {path}')
        rows = load_json(path)
        lookup = build_lookup(rows)
        baseline_rows[method] = lookup
        baseline_user_sets.append(set(lookup.keys()))

    selected = pick_five_users(gt_lookup, person_lookup, hh_lookup, baseline_user_sets)
    selected_ids = [x['user_id'] for x in selected]

    gt_selected = [gt_lookup[uid] for uid in selected_ids]

    ca_seqs, ca_raw = rbf.load_schedules(str(GT_FILE))
    n_train = int(len(ca_seqs) * 0.8)
    train_seqs = ca_seqs[:n_train]

    mc = rbf.MarkovChainBaseline()
    mc.fit(train_seqs)
    freq = rbf.FrequencyBaseline()
    freq.fit(train_seqs)
    rule = rbf.RuleBasedHHBaseline(random_state=42)
    rule.fit(train_seqs, person_lookup=person_lookup, hh_lookup=hh_lookup)

    np.random.seed(42)
    mc_gen = mc.generate(len(selected_ids))
    np.random.seed(42)
    freq_gen = freq.generate(len(selected_ids))
    rule_rows = rule.generate_for_users(selected_ids)

    synthetic_rows = {
        'MarkovChain': [{'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(selected_ids, mc_gen)],
        'Empirical Sampling': [{'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(selected_ids, freq_gen)],
        'Rule-based (CDAP)': rule_rows,
    }

    all_method_rows = {}
    all_method_rows.update(synthetic_rows)
    for method, lookup in baseline_rows.items():
        all_method_rows[method] = [lookup[uid] for uid in selected_ids]

    metrics_rows = []
    acc_heat = {}
    for method, pred_rows in all_method_rows.items():
        res = eval_pack(pred_rows, gt_selected)
        row = {'method': method}
        for m in METRICS:
            row[m] = float(res[m])
        metrics_rows.append(row)
        acc_heat[method] = per_user_accuracy(pred_rows, gt_selected)

    metrics_df = pd.DataFrame(metrics_rows)
    order = [
        'DeepMove', 'LSTPM', 'MarkovChain', 'Empirical Sampling', 'Rule-based (CDAP)',
        'Indiv-Base', 'Indiv-CoPB', 'HH-Base', 'HH-RAG', 'HoMe-LLM (Ours)'
    ]
    metrics_df['method'] = pd.Categorical(metrics_df['method'], categories=order, ordered=True)
    metrics_df = metrics_df.sort_values('method').reset_index(drop=True)

    user_df = pd.DataFrame(selected)
    user_df.to_csv(OUT_DIR / 'five_user_selected_cases.csv', index=False)
    metrics_df.to_csv(OUT_DIR / 'five_user_all_baselines_metrics.csv', index=False)

    labels_high = ['accuracy', 'bleu_score']
    labels_low_a = ['edit_dist', 'micro_hour', 'macro_int']
    labels_low_b = ['data_jsd', 'act_type', 'traj_len']

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    x = np.arange(len(metrics_df))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w / 2, metrics_df['accuracy'], width=w, label='Acc ↑', color='#4C78A8')
    ax.bar(x + w / 2, metrics_df['bleu_score'], width=w, label='BLEU ↑', color='#72B7B2')
    ax.set_title('Sequence Similarity (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['method'], rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, metrics_df['edit_dist'], marker='o', label='EditDist ↓', color='#F58518')
    ax.plot(x, metrics_df['micro_hour'], marker='s', label='Hour(micro) ↓', color='#E45756')
    ax.plot(x, metrics_df['macro_int'], marker='^', label='Interval(macro) ↓', color='#B279A2')
    ax.set_title('Temporal Alignment (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['method'], rotation=30, ha='right')
    ax.set_ylim(0, max(metrics_df[labels_low_a].max()) * 1.15)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(x, metrics_df['data_jsd'], marker='o', label='Data JSD ↓', color='#54A24B')
    ax.plot(x, metrics_df['act_type'], marker='s', label='ActType ↓', color='#EECA3B')
    ax.plot(x, metrics_df['traj_len'], marker='^', label='TrajLen ↓', color='#FF9DA6')
    ax.set_title('Distributional Consistency (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['method'], rotation=30, ha='right')
    ax.set_ylim(0, max(metrics_df[labels_low_b].max()) * 1.15)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'five_user_all_baselines_overview.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUT_DIR / 'five_user_all_baselines_overview.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    heat_rows = []
    for method in order:
        vals = acc_heat.get(method)
        if vals is None:
            continue
        for i, uid in enumerate(selected_ids):
            heat_rows.append({'method': method, 'user_id': uid, 'accuracy': vals[i]})
    heat_df = pd.DataFrame(heat_rows)
    heat_pivot = heat_df.pivot(index='method', columns='user_id', values='accuracy').reindex(order)

    plt.figure(figsize=(9, 6))
    sns.heatmap(heat_pivot, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
    plt.title('Per-user Accuracy Across All Baselines (5 selected users)')
    plt.xlabel('Selected User ID')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'five_user_all_baselines_accuracy_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUT_DIR / 'five_user_all_baselines_accuracy_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    print('Selected users:')
    print(user_df[['user_id', 'household_id', 'household_size', 'completeness', 'richness']].to_string(index=False))
    print('\nMetrics summary (5 users):')
    print(metrics_df.to_string(index=False))
    print(f"\nSaved CSV: {OUT_DIR / 'five_user_all_baselines_metrics.csv'}")
    print(f"Saved CSV: {OUT_DIR / 'five_user_selected_cases.csv'}")
    print(f"Saved figure: {OUT_DIR / 'five_user_all_baselines_overview.pdf'}")
    print(f"Saved figure: {OUT_DIR / 'five_user_all_baselines_accuracy_heatmap.pdf'}")


if __name__ == '__main__':
    run()
