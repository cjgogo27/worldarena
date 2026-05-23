import json
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PERSON_FILE = '/data/alice/cjtest/FinalTraj/California/processed_data/california_person_static.json'
HH_FILE = '/data/alice/cjtest/FinalTraj/California/processed_data/california_household_static.json'
GT_FILE = '/data/alice/cjtest/FinalTraj/California/processed_data/all_user_schedules.json'
GEN_FILE = '/data/alice/cjtest/FinalTraj/Trajectory_Generation_multi_agent/output_trajectories/all_trajectories_20251214_180155_California.json'
FIG_DIR = '/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/fig'
RESULT_CSV = '/data/alice/cjtest/FinalTraj/evaluation/gender_bias_results.csv'

ACTIVITY_ORDER = [
    'home', 'work', 'education', 'shopping', 'service',
    'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup'
]

STEREOTYPICAL_FEMALE = ['shopping', 'dropoff_pickup', 'medical']
STEREOTYPICAL_MALE = ['work', 'socialize', 'exercise']


def time_to_minutes(time_str):
    if time_str == '24:00':
        return 1440
    h, m = time_str.split(':')
    return int(h) * 60 + int(m)


def schedule_to_counts(schedule):
    counts = dict((a, 0.0) for a in ACTIVITY_ORDER)
    for seg in schedule:
        act = seg.get('activity', 'home')
        if act not in counts:
            continue
        start = time_to_minutes(seg.get('start_time', '00:00'))
        end = time_to_minutes(seg.get('end_time', '24:00'))
        if end > start:
            counts[act] += float(end - start)
    return counts


def load_gender_lookup(person_file):
    with open(person_file) as f:
        people = json.load(f)
    lookup = {}
    for p in people:
        uid = str(p.get('user_id', ''))
        g = str(p.get('gender', '')).strip().capitalize()
        if g in {'Male', 'Female'} and uid:
            lookup[uid] = g
    return lookup


def aggregate_gender_fractions(data, gender_lookup):
    totals = {
        'Male': dict((a, 0.0) for a in ACTIVITY_ORDER),
        'Female': dict((a, 0.0) for a in ACTIVITY_ORDER),
    }
    total_minutes = {'Male': 0.0, 'Female': 0.0}
    user_counts = {'Male': 0, 'Female': 0}

    for item in data:
        uid = str(item.get('user_id', ''))
        g = gender_lookup.get(uid)
        if g not in {'Male', 'Female'}:
            continue
        c = schedule_to_counts(item.get('schedule', []))
        day_total = sum(c.values())
        if day_total <= 0:
            continue
        user_counts[g] += 1
        total_minutes[g] += day_total
        for act in ACTIVITY_ORDER:
            totals[g][act] += c[act]

    fractions = {'Male': {}, 'Female': {}}
    for g in ['Male', 'Female']:
        denom = total_minutes[g] if total_minutes[g] > 0 else 1.0
        for act in ACTIVITY_ORDER:
            fractions[g][act] = totals[g][act] / denom

    return fractions, totals, user_counts


def chi_square_generated(totals):
    contingency = np.array([
        [totals['Male'][act] for act in ACTIVITY_ORDER],
        [totals['Female'][act] for act in ACTIVITY_ORDER],
    ], dtype=float)
    contingency = contingency + 1e-10

    def _chi2_stat(table):
        row_sum = table.sum(axis=1, keepdims=True)
        col_sum = table.sum(axis=0, keepdims=True)
        total = table.sum()
        expected = (row_sum @ col_sum) / (total + 1e-12)
        return float(np.sum((table - expected) ** 2 / (expected + 1e-12)))

    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(contingency)
    except Exception:
        chi2 = _chi2_stat(contingency)
        dof = int((contingency.shape[0] - 1) * (contingency.shape[1] - 1))
        pooled = contingency.sum(axis=0)
        pooled = pooled / pooled.sum()
        n_male = int(round(contingency[0].sum()))
        n_female = int(round(contingency[1].sum()))
        rng = np.random.default_rng(42)
        sim_stats = []
        for _ in range(2000):
            sim_table = np.vstack([
                rng.multinomial(n_male, pooled),
                rng.multinomial(n_female, pooled),
            ]).astype(float)
            sim_stats.append(_chi2_stat(sim_table))
        sim_stats = np.array(sim_stats)
        p = float((np.sum(sim_stats >= chi2) + 1) / (len(sim_stats) + 1))

    return chi2, p, dof


def run_analysis():
    os.makedirs(FIG_DIR, exist_ok=True)

    gender_lookup = load_gender_lookup(PERSON_FILE)

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    with open(GEN_FILE) as f:
        gen_data = json.load(f)

    gt_frac, gt_totals, gt_user_counts = aggregate_gender_fractions(gt_data, gender_lookup)
    gen_frac, gen_totals, gen_user_counts = aggregate_gender_fractions(gen_data, gender_lookup)

    if min(gen_user_counts['Male'], gen_user_counts['Female']) < 10:
        print('NOTE: fewer than 10 users in at least one generated-data gender group; interpret with caution.')

    rows = []
    for act in ACTIVITY_ORDER:
        gt_gap = abs(gt_frac['Female'][act] - gt_frac['Male'][act])
        gen_gap = abs(gen_frac['Female'][act] - gen_frac['Male'][act])
        amp = gen_gap - gt_gap
        ratio = gen_gap / (gt_gap + 1e-10)
        rows.append({
            'activity': act,
            'gt_male': gt_frac['Male'][act],
            'gt_female': gt_frac['Female'][act],
            'gen_male': gen_frac['Male'][act],
            'gen_female': gen_frac['Female'][act],
            'gt_gender_gap_abs': gt_gap,
            'gen_gender_gap_abs': gen_gap,
            'bias_amplification': amp,
            'gap_ratio_gen_over_gt': ratio,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)

    chi2, p, dof = chi_square_generated(gen_totals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [2.2, 1.3]})

    x = np.arange(len(ACTIVITY_ORDER))
    w = 0.2
    ax1.bar(x - 1.5 * w, [gt_frac['Male'][a] for a in ACTIVITY_ORDER], width=w, label='GT-Male', color='#4E79A7')
    ax1.bar(x - 0.5 * w, [gt_frac['Female'][a] for a in ACTIVITY_ORDER], width=w, label='GT-Female', color='#F28E2B')
    ax1.bar(x + 0.5 * w, [gen_frac['Male'][a] for a in ACTIVITY_ORDER], width=w, label='Gen-Male', color='#59A14F')
    ax1.bar(x + 1.5 * w, [gen_frac['Female'][a] for a in ACTIVITY_ORDER], width=w, label='Gen-Female', color='#E15759')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace('_', '\n') for a in ACTIVITY_ORDER], fontsize=10)
    ax1.set_ylabel('Time Fraction', fontsize=12)
    ax1.set_title('A. Gender-conditioned Activity Time Fractions (Ground Truth vs Generated)', fontsize=14, fontweight='bold')
    ax1.legend(ncol=4, fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    target_acts = STEREOTYPICAL_FEMALE + STEREOTYPICAL_MALE
    x2 = np.arange(len(target_acts))
    gt_gaps = [abs(gt_frac['Female'][a] - gt_frac['Male'][a]) for a in target_acts]
    gen_gaps = [abs(gen_frac['Female'][a] - gen_frac['Male'][a]) for a in target_acts]
    amp_vals = [g2 - g1 for g1, g2 in zip(gt_gaps, gen_gaps)]

    ax2.bar(x2 - 0.15, gt_gaps, width=0.3, label='GT Gap |F-M|', color='#9C755F')
    ax2.bar(x2 + 0.15, gen_gaps, width=0.3, label='Generated Gap |F-M|', color='#B07AA1')
    for i, v in enumerate(amp_vals):
        ax2.text(i, max(gt_gaps[i], gen_gaps[i]) + 0.001, f'+{v:.3f}' if v >= 0 else f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([a.replace('_', '\n') for a in target_acts], fontsize=10)
    ax2.set_ylabel('Absolute Gender Gap', fontsize=12)
    ax2.set_title('B. Bias Amplification on Stereotypical Activities', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig_pdf = os.path.join(FIG_DIR, 'gender_bias_analysis.pdf')
    fig_png = os.path.join(FIG_DIR, 'gender_bias_analysis.png')
    plt.savefig(fig_pdf, bbox_inches='tight', dpi=220)
    plt.savefig(fig_png, bbox_inches='tight', dpi=220)
    plt.close()

    print('=== Gender Bias Analysis ===')
    print(f'GT users with gender: Male={gt_user_counts["Male"]}, Female={gt_user_counts["Female"]}')
    print(f'Generated users with gender: Male={gen_user_counts["Male"]}, Female={gen_user_counts["Female"]}')
    print(f'Chi-square test (generated male vs female activity distribution): chi2={chi2:.4f}, dof={dof}, p={p}')

    ranked = df.sort_values('bias_amplification', ascending=False)
    print('Top bias amplification activities:')
    for _, r in ranked.head(6).iterrows():
        print(f"  {r['activity']}: amplification={r['bias_amplification']:.4f}, gt_gap={r['gt_gender_gap_abs']:.4f}, gen_gap={r['gen_gender_gap_abs']:.4f}")

    print(f'Saved figure: {fig_pdf}')
    print(f'Saved figure: {fig_png}')
    print(f'Saved table:  {RESULT_CSV}')


if __name__ == '__main__':
    run_analysis()
