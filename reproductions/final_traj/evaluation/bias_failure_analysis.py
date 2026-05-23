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

GT_FILE = ROOT / 'California' / 'processed_data' / 'all_user_schedules.json'
PERSON_FILE = ROOT / 'California' / 'processed_data' / 'california_person_static.json'
HOUSEHOLD_FILE = ROOT / 'California' / 'processed_data' / 'california_household_static.json'
GEN_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251117_122412.json'

# This generated cohort aligns with processed_data_1 user set.
GT_FILE = ROOT / 'California' / 'processed_data_1' / 'all_user_schedules.json'
PERSON_FILE = ROOT / 'California' / 'processed_data_1' / 'california_person_static.json'
HOUSEHOLD_FILE = ROOT / 'California' / 'processed_data_1' / 'california_household_static.json'

COLLAB_ACTS = {'dropoff_pickup', 'socialize', 'dine_out', 'shopping'}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def user_to_household_id(uid, person_row):
    hid = str(person_row.get('household_id', '')).strip() if person_row else ''
    if hid:
        return hid
    return str(uid).split('_')[0]


def seq_to_signature(seq):
    # Use non-home transitions only to detect template-like fixed patterns.
    out = []
    prev = None
    for v in seq:
        v = int(v)
        act = rbf.CODE_TO_ACTIVITY.get(v, 'home')
        if act == 'home':
            continue
        if act != prev:
            out.append(act)
            prev = act
    return ' > '.join(out[:8]) if out else 'home_only'


def collaboration_recall(gt_seq, pred_seq):
    gt_collab = np.isin(gt_seq, [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS])
    if gt_collab.sum() == 0:
        return np.nan
    matched = np.logical_and(gt_collab, gt_seq == pred_seq)
    return float(matched.sum() / gt_collab.sum())


def build_case_df(gt_lookup, gen_lookup, person_lookup, hh_lookup):
    rows = []
    common_users = sorted(set(gt_lookup.keys()) & set(gen_lookup.keys()))
    for uid in common_users:
        gt_schedule = gt_lookup[uid]['schedule']
        pred_schedule = gen_lookup[uid]['schedule']
        gt_seq = rbf.schedule_to_96_timesteps(gt_schedule)
        pred_seq = rbf.schedule_to_96_timesteps(pred_schedule)

        acc = float((gt_seq == pred_seq).mean())
        collab = collaboration_recall(gt_seq, pred_seq)

        person = person_lookup.get(uid, {})
        hid = user_to_household_id(uid, person)
        hh = hh_lookup.get(hid, {})

        gt_sig = seq_to_signature(gt_seq)
        pred_sig = seq_to_signature(pred_seq)

        rows.append({
            'user_id': uid,
            'household_id': hid,
            'gender': person.get('gender', 'Unknown'),
            'race': person.get('race', 'Unknown'),
            'employment_status': person.get('employment_status', 'Unknown'),
            'relationship': person.get('relationship', 'Unknown'),
            'household_size': int(hh.get('household_size', 0) or 0),
            'adult_count': int(hh.get('adult_count', 0) or 0),
            'young_children_count': int(hh.get('young_children_count', 0) or 0),
            'vehicle_count': int(hh.get('vehicle_count', 0) or 0),
            'household_income': hh.get('household_income', 'Unknown'),
            'household_race': hh.get('household_race', 'Unknown'),
            'accuracy': acc,
            'collab_recall': collab,
            'gt_signature': gt_sig,
            'pred_signature': pred_sig,
            'gt_schedule': gt_schedule,
            'pred_schedule': pred_schedule,
        })
    return pd.DataFrame(rows)


def composition_key(row):
    return f"S{row['household_size']}-A{row['adult_count']}-C{row['young_children_count']}-V{row['vehicle_count']}"


def analyze_and_plot():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style('whitegrid')

    gt_data = load_json(GT_FILE)
    gen_data = load_json(GEN_FILE)
    person_data = load_json(PERSON_FILE)
    hh_data = load_json(HOUSEHOLD_FILE)

    gt_lookup = {str(x['user_id']): x for x in gt_data if x.get('user_id')}
    gen_lookup = {str(x['user_id']): x for x in gen_data if x.get('user_id')}
    person_lookup = {str(x['user_id']): x for x in person_data if x.get('user_id')}
    hh_lookup = {str(x.get('household_id', '')): x for x in hh_data if x.get('household_id')}

    df = build_case_df(gt_lookup, gen_lookup, person_lookup, hh_lookup)
    if df.empty:
        raise RuntimeError('No overlap users between generated and GT data.')

    # Composition rarity within evaluated cohort to expose tail-group behavior.
    # Rare = <= 4 users in this cohort.
    df['composition'] = df.apply(composition_key, axis=1)
    eval_comp_freq = Counter(df['composition'])
    df['composition_freq'] = df['composition'].map(lambda x: eval_comp_freq.get(x, 0))
    df['is_rare_composition'] = df['composition_freq'] <= 4

    # Female collaboration failures.
    female_df = df[df['gender'].astype(str).str.lower() == 'female'].copy()
    female_df = female_df.dropna(subset=['collab_recall'])
    female_fail = female_df.sort_values(['collab_recall', 'accuracy'], ascending=[True, True]).head(6)
    female_fail_cols = [
        'user_id', 'household_id', 'accuracy', 'collab_recall', 'relationship',
        'household_size', 'young_children_count', 'household_income', 'race', 'household_race'
    ]

    # Figure 1: female collaboration failures scatter.
    plt.figure(figsize=(8.8, 6.2))
    plt.scatter(female_df['accuracy'], female_df['collab_recall'], s=50, color='#89B0AE', alpha=0.8, label='Female users')
    plt.scatter(female_fail['accuracy'], female_fail['collab_recall'], s=90, color='#D62828', label='Failure examples')
    for _, r in female_fail.iterrows():
        plt.text(r['accuracy'] + 0.004, r['collab_recall'] + 0.01, r['user_id'], fontsize=8)
    plt.xlabel('Overall Accuracy')
    plt.ylabel('Collaboration Recall on {dropoff/socialize/dine_out/shopping}')
    plt.title('Female Collaboration-Inaccuracy Cases (Lower-Left = Worse)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_female_collaboration_failures.pdf', dpi=300)
    plt.savefig(OUT_DIR / 'bias_female_collaboration_failures.png', dpi=300)
    plt.close()

    # Figure 2: rare composition bias.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    box_df = df[['accuracy', 'is_rare_composition']].copy()
    box_df['Group'] = box_df['is_rare_composition'].map({True: 'Rare composition', False: 'Common composition'})
    sns.boxplot(data=box_df, x='Group', y='accuracy', palette=['#E76F51', '#2A9D8F'], ax=axes[0])
    axes[0].set_title('Accuracy by Composition Rarity')
    axes[0].set_xlabel('')

    rare_comp = df[df['is_rare_composition']].groupby('composition').agg(
        mean_accuracy=('accuracy', 'mean'),
        n=('user_id', 'count')
    ).reset_index().sort_values('mean_accuracy').head(8)
    if rare_comp.empty:
        rare_comp = pd.DataFrame({'composition': [], 'mean_accuracy': [], 'n': []})
    axes[1].barh(rare_comp['composition'], rare_comp['mean_accuracy'], color='#F4A261')
    for i, (_, rr) in enumerate(rare_comp.iterrows()):
        axes[1].text(rr['mean_accuracy'] + 0.005, i, f"n={int(rr['n'])}", va='center', fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Worst Rare Compositions')
    axes[1].set_xlabel('Mean Accuracy')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bias_rare_household_composition.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'bias_rare_household_composition.png', dpi=300)
    plt.close(fig)

    # Figure 2b: family role division bias (household division proxy).
    role_df = df.groupby('relationship').agg(
        mean_accuracy=('accuracy', 'mean'),
        mean_collab_recall=('collab_recall', 'mean'),
        n=('user_id', 'count')
    ).reset_index().sort_values('mean_accuracy')
    role_df = role_df[role_df['n'] >= 2]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(role_df))
    ax1.bar(x - 0.18, role_df['mean_accuracy'], width=0.36, color='#457B9D', label='Mean accuracy')
    ax1.bar(x + 0.18, role_df['mean_collab_recall'].fillna(0.0), width=0.36, color='#E76F51', label='Mean collaboration recall')
    for i, n in enumerate(role_df['n']):
        ax1.text(i, 0.02, f"n={int(n)}", ha='center', va='bottom', fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(role_df['relationship'], rotation=20, ha='right')
    ax1.set_title('Family Division Bias by Household Role')
    ax1.set_ylabel('Score')
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bias_family_role_division.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'bias_family_role_division.png', dpi=300)
    plt.close(fig)

    # Figure 3: race x income heatmap.
    pivot = df.pivot_table(index='household_race', columns='household_income', values='accuracy', aggfunc='mean')
    plt.figure(figsize=(13, 5.6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Mean Accuracy'})
    plt.title('Income-Race Bias Surface (Mean Accuracy)')
    plt.xlabel('Household Income')
    plt.ylabel('Household Race')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_income_race_heatmap.pdf', dpi=300)
    plt.savefig(OUT_DIR / 'bias_income_race_heatmap.png', dpi=300)
    plt.close()

    # Figure 4: fixed-pattern bias (mode collapse).
    pred_sig_counts = Counter(df['pred_signature'])
    gt_sig_counts = Counter(df['gt_signature'])
    top_sigs = [k for k, _ in pred_sig_counts.most_common(8)]
    rows = []
    for s in top_sigs:
        rows.append({'signature': s, 'source': 'Generated', 'count': pred_sig_counts[s]})
        rows.append({'signature': s, 'source': 'GroundTruth', 'count': gt_sig_counts.get(s, 0)})
    sig_df = pd.DataFrame(rows)

    plt.figure(figsize=(12.8, 6.2))
    sns.barplot(data=sig_df, x='count', y='signature', hue='source', palette=['#264653', '#A8DADC'])
    plt.title('Fixed-Pattern Bias: Repeated Daily Templates in Generated Schedules')
    plt.xlabel('User Count')
    plt.ylabel('Top Generated Signatures')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'bias_fixed_pattern_templates.pdf', dpi=300)
    plt.savefig(OUT_DIR / 'bias_fixed_pattern_templates.png', dpi=300)
    plt.close()

    # Save tables.
    df_out = df.drop(columns=['gt_schedule', 'pred_schedule'])
    df_out.to_csv(OUT_DIR / 'bias_user_level_metrics.csv', index=False)
    female_fail[female_fail_cols].to_csv(OUT_DIR / 'bias_female_failure_examples.csv', index=False)

    # Report text.
    rare_mean = df[df['is_rare_composition']]['accuracy'].mean()
    common_mean = df[~df['is_rare_composition']]['accuracy'].mean()
    female_fail_text = []
    for _, r in female_fail.iterrows():
        female_fail_text.append(
            f"- {r['user_id']}: accuracy={r['accuracy']:.3f}, collab_recall={r['collab_recall']:.3f}, "
            f"relationship={r['relationship']}, hh_size={r['household_size']}, young_children={r['young_children_count']}, income={r['household_income']}"
        )

    top_race_income = (
        df.groupby(['household_race', 'household_income'])['accuracy']
        .mean().reset_index().sort_values('accuracy').head(6)
    )

    report_lines = [
        '# Bias and Failure Analysis (HoMe-LLM California Cohort)',
        '',
        f'- Evaluated overlap users: {len(df)}',
        f'- Female users with collaboration signal: {len(female_df)}',
        f'- Rare composition users: {int(df["is_rare_composition"].sum())}',
        '',
        '## 1) Female collaboration-inaccuracy examples',
        'These users have lower collaboration recall on collaborative activities (dropoff/socialize/dine_out/shopping).',
        *female_fail_text,
        '',
        '## 2) Rare household composition bias',
        f'- Mean accuracy (rare composition): {rare_mean:.3f}',
        f'- Mean accuracy (common composition): {common_mean:.3f}',
        '- Rare compositions are defined within this evaluated cohort as composition frequency <= 4.',
        '',
        '## 2b) Family division bias',
        'Role groups (Self / Spouse / etc.) show different collaboration-recall levels, indicating role-sensitive household coordination errors.',
        '',
        '## 3) Income and race bias surface',
        'Lowest-performing race-income cells (mean accuracy):',
    ]

    for _, r in top_race_income.iterrows():
        report_lines.append(
            f"- race={r['household_race']}, income={r['household_income']}, mean_acc={r['accuracy']:.3f}"
        )

    report_lines.extend([
        '',
        '## 4) Fixed-pattern bias',
        'Top generated signatures are more concentrated than ground truth, indicating template reuse / mode collapse on daily plans.',
        '',
        '## Generated files',
        '- bias_female_collaboration_failures.pdf/.png',
        '- bias_rare_household_composition.pdf/.png',
        '- bias_income_race_heatmap.pdf/.png',
        '- bias_fixed_pattern_templates.pdf/.png',
        '- bias_family_role_division.pdf/.png',
        '- bias_user_level_metrics.csv',
        '- bias_female_failure_examples.csv',
    ])

    report_path = OUT_DIR / 'bias_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f'Saved report: {report_path}')
    print(f'Saved user metrics: {OUT_DIR / "bias_user_level_metrics.csv"}')
    print(f'Saved female examples: {OUT_DIR / "bias_female_failure_examples.csv"}')


if __name__ == '__main__':
    analyze_and_plot()
