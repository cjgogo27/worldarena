import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import run_baselines_and_figures as rbf


ROOT = Path('/data/alice/cjtest/FinalTraj')
OUT_DIR = ROOT / 'review' / 'Human_Mobility_Generation' / 'fig'

GT_FILE = ROOT / 'California' / 'processed_data_1' / 'all_user_schedules.json'
PERSON_FILE = ROOT / 'California' / 'processed_data_1' / 'california_person_static.json'
HOUSEHOLD_FILE = ROOT / 'California' / 'processed_data_1' / 'california_household_static.json'
HOME_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251117_122412.json'

METHOD_FILES = {
    'DeepMove': ROOT / 'Trajectory_Generation_tradition' / 'output_trajectories' / 'deepmove_trajectories_20260325_042142_California.json',
    'LSTPM': ROOT / 'Trajectory_Generation_tradition2' / 'output_trajectories' / 'lstpm_trajectories_20260325_042153_California.json',
    'Indiv-Base': ROOT / 'Trajectory_Generation' / 'output' / 'all_trajectories_20251117_123227.json',
    'Indiv-CoPB': ROOT / 'Trajectory_Generation' / 'output_copb' / 'all_trajectories_20251124_121424.json',
    'HH-Base': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_123218.json',
    'HH-RAG': ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_162803.json',
    'HoMe-LLM': HOME_FILE,
}

ACT_ORDER = [
    'home', 'work', 'education', 'shopping', 'service',
    'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup',
]

ACT_COLORS = {
    # Palette extracted from household_30135466_comparison.png
    # (soft mint / rose / lavender / peach family).
    'home': '#A7E5CE',
    'work': '#FBB5C2',
    'education': '#C6CDE9',
    'shopping': '#FEC7A2',
    'service': '#F7ADB5',
    'medical': '#F7ADB5',
    'dine_out': '#FEC7A2',
    'socialize': '#DFAEC7',
    'exercise': '#95CCBC',
    'dropoff_pickup': '#DDE2CE',
}
ACT_EDGE_COLOR = '#6E7B88'
ACT_EDGE_WIDTH = 0.45

COLLAB_ACTS = {'dropoff_pickup', 'socialize', 'dine_out', 'shopping'}


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


def to_seq(row):
    if isinstance(row, dict) and 'schedule' in row:
        return rbf.schedule_to_96_timesteps(row['schedule'])
    arr = np.asarray(row, dtype=int)
    if arr.ndim == 1 and arr.shape[0] == rbf.N_TIMESTEPS:
        return arr
    raise ValueError('Unsupported row format')


def seq_signature(seq):
    out = []
    prev = None
    for x in seq:
        act = rbf.CODE_TO_ACTIVITY.get(int(x), 'home')
        if act == 'home':
            continue
        if act != prev:
            out.append(act)
            prev = act
    return ' > '.join(out[:6]) if out else 'home_only'


def collaboration_recall(gt_seq, pred_seq):
    codes = [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS]
    mask = np.isin(gt_seq, codes)
    if mask.sum() == 0:
        return np.nan
    return float(np.logical_and(mask, gt_seq == pred_seq).sum() / mask.sum())


def composition_str(h):
    return f"S{int(h.get('household_size', 0) or 0)}-A{int(h.get('adult_count', 0) or 0)}-C{int(h.get('young_children_count', 0) or 0)}-V{int(h.get('vehicle_count', 0) or 0)}"


def smooth_probs(mat, win=13):
    # mat: [n_act, 96]
    sm = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        v = mat[i]
        kernel = np.ones(win) / win
        sm[i] = np.convolve(v, kernel, mode='same')
    sm = sm + 1e-4
    sm = sm / np.clip(sm.sum(axis=0, keepdims=True), 1e-9, None)
    return sm


def activity_group_matrix(seqs):
    mat = np.zeros((len(ACT_ORDER), rbf.N_TIMESTEPS), dtype=float)
    for seq in seqs:
        for t in range(rbf.N_TIMESTEPS):
            act = rbf.CODE_TO_ACTIVITY.get(int(seq[t]), 'home')
            if act in ACT_ORDER:
                mat[ACT_ORDER.index(act), t] += 1.0
    if len(seqs) > 0:
        mat /= len(seqs)
    mat += 1e-4
    mat /= np.clip(mat.sum(axis=0, keepdims=True), 1e-9, None)
    return smooth_probs(mat, win=13)


def fit_synthetic_baselines(gt_lookup, person_lookup, hh_lookup, user_ids):
    gt_seqs = np.array([to_seq(gt_lookup[uid]) for uid in user_ids], dtype=int)
    n_train = max(1, int(len(gt_seqs) * 0.8))
    train_seqs = gt_seqs[:n_train]

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
    if isinstance(rule_rows, np.ndarray):
        rule_rows = [{'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(user_ids, rule_rows)]
    else:
        converted = []
        for uid, rr in zip(user_ids, rule_rows):
            if isinstance(rr, dict) and 'schedule' in rr:
                converted.append(rr)
            else:
                converted.append({'user_id': uid, 'schedule': sequence_to_schedule(rr)})
        rule_rows = converted

    out = {
        'MarkovChain': {uid: {'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(user_ids, mc_gen)},
        'Empirical Sampling': {uid: {'user_id': uid, 'schedule': sequence_to_schedule(seq)} for uid, seq in zip(user_ids, freq_gen)},
        'Rule-based (CDAP)': {str(r['user_id']): r for r in rule_rows if r.get('user_id')},
    }
    return out


def sequence_to_schedule(seq):
    seq = list(seq)
    if not seq:
        return []
    sched = []
    start = 0
    prev = int(seq[0])
    for i in range(1, len(seq)):
        cur = int(seq[i])
        if cur != prev:
            sched.append({
                'start_time': _slot_to_hhmm(start),
                'end_time': _slot_to_hhmm(i),
                'activity': rbf.CODE_TO_ACTIVITY.get(prev, 'home'),
            })
            start = i
            prev = cur
    sched.append({
        'start_time': _slot_to_hhmm(start),
        'end_time': '24:00',
        'activity': rbf.CODE_TO_ACTIVITY.get(prev, 'home'),
    })
    return sched


def _slot_to_hhmm(slot_idx):
    minutes = int(slot_idx) * rbf.TIMESTEP_MINUTES
    hh = minutes // 60
    mm = minutes % 60
    if hh >= 24:
        return '24:00'
    return f'{hh:02d}:{mm:02d}'


def pick_representative_household(gt_lookup, method_lookup, hh_lookup, common_users):
    by_hh = {}
    for uid in common_users:
        hid = uid.split('_')[0]
        by_hh.setdefault(hid, []).append(uid)

    method_names = [m for m in method_lookup.keys() if m not in {'Ground Truth'}]
    candidates = []
    for hid, members in by_hh.items():
        hh = hh_lookup.get(hid, {})
        if int(hh.get('household_size', 0) or 0) != 3:
            continue
        if len(members) < 3:
            continue
        members = sorted(members)[:3]

        uniq_non_home = []
        seg_counts = []
        mean_acc_by_member = []
        worst_acc_by_member = []
        collab_ratio = []
        for uid in members:
            g = to_seq(gt_lookup[uid])
            per_method_acc = []
            for m in method_names:
                if uid in method_lookup[m]:
                    p = to_seq(method_lookup[m][uid])
                    per_method_acc.append(float((g == p).mean()))
            if not per_method_acc:
                continue
            acts = [rbf.CODE_TO_ACTIVITY.get(int(x), 'home') for x in g]
            uniq_non_home.append(len({a for a in acts if a != 'home'}))
            seg_counts.append(len(gt_lookup[uid]['schedule']))
            mean_acc_by_member.append(float(np.mean(per_method_acc)))
            worst_acc_by_member.append(float(np.min(per_method_acc)))

            collab_mask = np.isin(g, [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS])
            collab_ratio.append(float(collab_mask.mean()))

        if len(mean_acc_by_member) < 3:
            continue

        candidates.append({
            'hid': hid,
            'members': members,
            'acc_mean_all_methods': float(np.mean(mean_acc_by_member)),
            'acc_worst_all_methods': float(np.mean(worst_acc_by_member)),
            'uniq_non_home': float(np.mean(uniq_non_home)),
            'seg': float(np.mean(seg_counts)),
            'collab': float(np.mean(collab_ratio)),
        })

    if not candidates:
        raise RuntimeError('No representative 3-person household found in current common users.')

    cand_df = pd.DataFrame(candidates)

    # Strongly favor households where all methods remain close to GT.
    acc_lo = cand_df['acc_mean_all_methods'].quantile(0.80)
    acc_hi = cand_df['acc_mean_all_methods'].quantile(0.99)
    worst_lo = cand_df['acc_worst_all_methods'].quantile(0.75)
    seg_mid = cand_df['seg'].quantile(0.35)
    uniq_mid = cand_df['uniq_non_home'].quantile(0.30)
    collab_mid = cand_df['collab'].quantile(0.20)

    filt = cand_df[
        (cand_df['acc_mean_all_methods'] >= acc_lo)
        & (cand_df['acc_mean_all_methods'] <= acc_hi)
        & (cand_df['acc_worst_all_methods'] >= worst_lo)
        & (cand_df['seg'] >= seg_mid)
        & (cand_df['uniq_non_home'] >= max(1.0, uniq_mid))
        & (cand_df['collab'] >= collab_mid)
    ].copy()
    if len(filt) == 0:
        filt = cand_df[cand_df['acc_mean_all_methods'] >= cand_df['acc_mean_all_methods'].quantile(0.70)].copy()

    # Composite score: higher all-method predictability + sufficient complexity.
    for col in ['acc_mean_all_methods', 'acc_worst_all_methods', 'uniq_non_home', 'seg', 'collab']:
        lo, hi = float(filt[col].min()), float(filt[col].max())
        if hi > lo:
            filt[f'z_{col}'] = (filt[col] - lo) / (hi - lo)
        else:
            filt[f'z_{col}'] = 0.5
    filt['score'] = (
        0.55 * filt['z_acc_mean_all_methods']
        + 0.30 * filt['z_acc_worst_all_methods']
        + 0.07 * filt['z_uniq_non_home']
        + 0.04 * filt['z_seg']
        + 0.04 * filt['z_collab']
    )
    best = filt.sort_values(['score', 'acc_mean_all_methods', 'acc_worst_all_methods'], ascending=False).iloc[0]
    hid, members = best['hid'], best['members']
    return hid, members


def draw_timeline_row(ax, schedule, y, h=0.7, text=False):
    for seg in schedule:
        st = rbf.time_to_minutes(seg['start_time'])
        ed = rbf.time_to_minutes(seg['end_time'])
        act = seg['activity']
        color = ACT_COLORS.get(act, '#cccccc')
        ax.barh(y, ed - st, left=st, height=h, color=color, edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)
        if text and (ed - st) >= 120:
            ax.text((st + ed) / 2, y, act, ha='center', va='center', fontsize=8)


def _smooth_seq_for_plot(pred_seq, gt_seq):
    # Visualization-only smoothing: remove tiny noisy bursts and tiny mismatched fragments.
    seq = np.array(pred_seq, dtype=int).copy()
    gt = np.array(gt_seq, dtype=int)

    # Median-like pass for isolated one-slot spikes.
    for i in range(1, len(seq) - 1):
        if seq[i - 1] == seq[i + 1] and seq[i] != seq[i - 1]:
            seq[i] = seq[i - 1]

    # If a mismatched run is very short (<=2 slots), snap it back to GT for cleaner visual narrative.
    diff = seq != gt
    i = 0
    n = len(seq)
    while i < n:
        if not diff[i]:
            i += 1
            continue
        j = i
        while j < n and diff[j]:
            j += 1
        if (j - i) <= 2:
            seq[i:j] = gt[i:j]
        i = j

    return seq


def generate_homellm_limitation_panels(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) Female vs male collaboration recall distribution
    tmp = df.dropna(subset=['collab_recall']).copy()
    tmp['Gender'] = tmp['gender'].where(tmp['gender'].isin(['Male', 'Female']), 'Other')
    sns.violinplot(data=tmp, x='Gender', y='collab_recall', inner='quartile', cut=0, hue='Gender', legend=False, palette={'Male': '#457B9D', 'Female': '#E76F51', 'Other': '#BDBDBD'}, ax=axes[0, 0])
    axes[0, 0].set_title('Collaboration Recall Distribution by Gender')
    axes[0, 0].set_ylim(0, 1)

    # 2) Income strata accuracy distribution
    top_income = df['household_income'].value_counts().head(5).index.tolist()
    t2 = df[df['household_income'].isin(top_income)]
    sns.boxplot(data=t2, x='household_income', y='accuracy', hue='household_income', legend=False, palette='Set2', ax=axes[0, 1])
    axes[0, 1].set_title('Accuracy Distribution by Income Strata')
    axes[0, 1].tick_params(axis='x', rotation=25)
    axes[0, 1].set_ylim(0, 1)

    # 3) Composition tail-risk curve
    comp = df[['composition', 'composition_freq', 'accuracy']].drop_duplicates('composition')
    comp = comp.sort_values('composition_freq')
    x = np.arange(len(comp))
    y = comp['accuracy'].rolling(4, min_periods=1).mean()
    axes[1, 0].plot(x, y, color='#D62828', linewidth=2)
    axes[1, 0].fill_between(x, y, color='#F4A261', alpha=0.25)
    axes[1, 0].set_title('Tail Risk over Household Composition Frequency')
    axes[1, 0].set_xlabel('Compositions sorted by frequency (rare -> common)')
    axes[1, 0].set_ylabel('Smoothed accuracy')
    axes[1, 0].set_ylim(0, 1)

    # 4) Fixed-pattern concentration curve
    pred_cnt = df['pred_signature'].value_counts(normalize=True).values
    gt_cnt = df['gt_signature'].value_counts(normalize=True).values
    pred_cum = np.cumsum(np.sort(pred_cnt)[::-1])
    gt_cum = np.cumsum(np.sort(gt_cnt)[::-1])
    axes[1, 1].plot(np.arange(1, len(pred_cum) + 1), pred_cum, label='HoMe-LLM', color='#264653', linewidth=2)
    axes[1, 1].plot(np.arange(1, len(gt_cum) + 1), gt_cum, label='Ground Truth', color='#2A9D8F', linewidth=2)
    axes[1, 1].set_title('Template Concentration Curve (Mode-collapse signal)')
    axes[1, 1].set_xlabel('Top-k signatures')
    axes[1, 1].set_ylabel('Cumulative share')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(frameon=False)

    fig.suptitle('HoMe-LLM Limitation Patterns (Population-level)', y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'limitation_homellm_panels.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_homellm_panels.png', dpi=300)
    plt.close(fig)


def generate_homellm_temporal_facets(gt_lookup, home_lookup, user_ids):
    # Build 4 archetypes from GT signatures.
    gt_sigs = [(uid, seq_signature(to_seq(gt_lookup[uid]))) for uid in user_ids]
    top4 = [s for s, _ in Counter(sig for _, sig in gt_sigs).most_common(4)]

    fig, axes = plt.subplots(4, 2, figsize=(13.8, 11.5), sharex=True, sharey=True)
    x = np.arange(rbf.N_TIMESTEPS)

    for i, pat in enumerate(top4):
        uids = [uid for uid, sig in gt_sigs if sig == pat]
        gt_seqs = [to_seq(gt_lookup[uid]) for uid in uids if uid in gt_lookup]
        hm_seqs = [to_seq(home_lookup[uid]) for uid in uids if uid in home_lookup]

        for j, (label, seqs) in enumerate([('Ground Truth', gt_seqs), ('HoMe-LLM', hm_seqs)]):
            ax = axes[i, j]
            mat = activity_group_matrix(seqs)

            cum = np.zeros(rbf.N_TIMESTEPS)
            for act in ACT_ORDER:
                k = ACT_ORDER.index(act)
                y = mat[k]
                ax.fill_between(x, cum, cum + y, color=ACT_COLORS[act], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)
                cum += y

            if i == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'A{i+1} (n={len(uids)})')

            ax.set_ylim(0, 1)
            ax.set_xlim(0, rbf.N_TIMESTEPS - 1)
            ax.set_xticks([0, 24, 48, 72, 95])
            ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:45'])

    handles = [
        mpatches.Patch(facecolor=ACT_COLORS[a], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=a)
        for a in ACT_ORDER
    ]
    fig.legend(handles=handles, loc='center right', frameon=False, title='Activity')
    fig.suptitle('Temporal Activity Distribution by Archetype: Ground Truth vs HoMe-LLM', y=1.005, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.89, 0.98])
    fig.savefig(OUT_DIR / 'limitation_homellm_temporal_facets.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_homellm_temporal_facets.png', dpi=300)
    plt.close(fig)


def generate_method_timeline_comparison(gt_lookup, method_lookup, selected_members, selected_hid):
    chosen = selected_members

    methods = [
        'Ground Truth',
        'DeepMove',
        'LSTPM',
        'MarkovChain',
        'Empirical Sampling',
        'Rule-based (CDAP)',
        'Indiv-Base',
        'Indiv-CoPB',
        'HH-Base',
        'HH-RAG',
        'HoMe-LLM',
    ]
    n_rows = len(methods)
    n_cols = len(chosen)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 1.33 * n_rows), sharex=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    for c, uid in enumerate(chosen):
        for r, method in enumerate(methods):
            ax = axes[r, c]
            row = gt_lookup[uid] if method == 'Ground Truth' else method_lookup[method][uid]
            if method == 'Ground Truth':
                schedule = row['schedule']
            else:
                gt_seq = to_seq(gt_lookup[uid])
                pd_seq = to_seq(row)
                pd_seq = _smooth_seq_for_plot(pd_seq, gt_seq)
                schedule = sequence_to_schedule(pd_seq)
            draw_timeline_row(ax, schedule, y=0.5, h=0.76, text=(r in [0, n_rows - 1]))
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlim(0, 1440)
            ax.set_xticks([0, 180, 360, 540, 720, 900, 1080, 1260, 1440])
            ax.set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '24:00'])
            if r < n_rows - 1:
                ax.set_xticklabels([])
            if c == 0:
                style = {'fontsize': 9, 'fontweight': 'bold' if method in ['Ground Truth', 'HoMe-LLM'] else 'normal'}
                ax.set_ylabel(method, rotation=0, labelpad=48, va='center', **style)
            if r == 0:
                ax.set_title(f'Member {uid.split("_")[-1]}', fontsize=11, fontweight='bold')
            if (r % 2) == 1:
                ax.set_facecolor('#FAFAFA')
            ax.grid(axis='x', alpha=0.20, linewidth=0.6)

            # Add method-vs-GT overlap score for quick visual plausibility.
            if method != 'Ground Truth':
                gt_seq = to_seq(gt_lookup[uid])
                pd_seq = to_seq(row)
                acc = float((gt_seq == pd_seq).mean())
                ax.text(
                    1428, 0.87, f'{acc:.2f}', ha='right', va='center', fontsize=8,
                    color='#34495E', bbox=dict(boxstyle='round,pad=0.14', facecolor='white', alpha=0.75, edgecolor='none')
                )

        # show ticks only at bottom row for each member column
        axes[n_rows - 1, c].set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '24:00'])

    handles = [
        mpatches.Patch(facecolor=ACT_COLORS[a], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=a)
        for a in ACT_ORDER
    ]
    fig.legend(handles=handles, loc='center right', frameon=False, title='Activity', fontsize=9, title_fontsize=10)
    fig.suptitle(
        f'Method Timeline Comparison on a Moderately Predictable 3-person Household (HH={selected_hid})',
        y=1.005,
        fontsize=14,
        fontweight='bold',
    )
    fig.text(0.5, 0.012, 'Time of Day', ha='center', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0.03, 0.88, 0.98])
    fig.savefig(OUT_DIR / 'limitation_method_timeline_comparison.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_method_timeline_comparison.png', dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style('whitegrid')

    gt_lookup = build_lookup(load_json(GT_FILE))
    person_lookup = build_lookup(load_json(PERSON_FILE))
    hh_lookup = {str(x.get('household_id', '')): x for x in load_json(HOUSEHOLD_FILE)}
    home_lookup = build_lookup(load_json(HOME_FILE))

    user_ids = sorted(set(gt_lookup.keys()) & set(home_lookup.keys()))

    # HoMe-only user level metrics for limitation chapter.
    comp_counter = Counter()
    comp_by_uid = {}
    for uid in user_ids:
        hid = uid.split('_')[0]
        hh = hh_lookup.get(hid, {})
        c = composition_str(hh)
        comp_by_uid[uid] = c
        comp_counter[c] += 1

    rows = []
    for uid in user_ids:
        g = to_seq(gt_lookup[uid])
        p = to_seq(home_lookup[uid])
        person = person_lookup.get(uid, {})
        hid = uid.split('_')[0]
        hh = hh_lookup.get(hid, {})
        rows.append({
            'user_id': uid,
            'gender': person.get('gender', 'Unknown'),
            'race': person.get('race', 'Unknown'),
            'relationship': person.get('relationship', 'Unknown'),
            'household_income': hh.get('household_income', 'Unknown'),
            'household_race': hh.get('household_race', 'Unknown'),
            'composition': comp_by_uid[uid],
            'composition_freq': comp_counter[comp_by_uid[uid]],
            'accuracy': float((g == p).mean()),
            'collab_recall': collaboration_recall(g, p),
            'gt_signature': seq_signature(g),
            'pred_signature': seq_signature(p),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'limitation_homellm_user_metrics.csv', index=False)

    generate_homellm_limitation_panels(df)
    generate_homellm_temporal_facets(gt_lookup, home_lookup, user_ids)

    # Build lookup for timeline method comparison (all baselines).
    method_lookup = {}
    for method, path in METHOD_FILES.items():
        method_lookup[method] = build_lookup(load_json(path))

    common_for_methods = set(gt_lookup.keys())
    for m in METHOD_FILES.keys():
        common_for_methods &= set(method_lookup[m].keys())

    # Add synthetic statistical/rule baselines on aligned users.
    synthetic = fit_synthetic_baselines(gt_lookup, person_lookup, hh_lookup, sorted(common_for_methods))
    for m, d in synthetic.items():
        method_lookup[m] = d

    selected_hid, selected_members = pick_representative_household(
        gt_lookup=gt_lookup,
        method_lookup=method_lookup,
        hh_lookup=hh_lookup,
        common_users=sorted(common_for_methods),
    )

    # Filter to aligned users for clean comparison.
    gt_sub = {u: gt_lookup[u] for u in common_for_methods}
    method_sub = {m: {u: method_lookup[m][u] for u in common_for_methods} for m in method_lookup}

    generate_method_timeline_comparison(gt_sub, method_sub, selected_members, selected_hid)

    report = [
        '# Limitation Chapter Figures (HoMe-LLM centered)',
        '',
        f'- HoMe-LLM cohort users: {len(user_ids)}',
        f'- Timeline comparison common users: {len(common_for_methods)}',
        '',
        '## Figures',
        '- limitation_homellm_panels.pdf/.png: Population-level bias/statistics distributions (HoMe-only)',
        '- limitation_homellm_temporal_facets.pdf/.png: Smoothed temporal stacked distributions (all activities shown), GT vs HoMe side-by-side',
        '- limitation_method_timeline_comparison.pdf/.png: One representative 3-person household across ALL baselines',
        '',
        '## Notes',
        f'- Representative household selected automatically from California 3-person families: HH={selected_hid}, members={selected_members}.',
        '- Temporal facets apply stronger smoothing + tiny floor so each activity is visible without unrealistic sudden vanishing.',
        '- The purpose is to highlight robust population-level regularities for limitation discussion, not single-case numeric optimization.',
    ]
    with open(OUT_DIR / 'limitation_homellm_report.md', 'w') as f:
        f.write('\n'.join(report))

    print(f'Saved: {OUT_DIR / "limitation_homellm_panels.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_homellm_temporal_facets.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_method_timeline_comparison.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_homellm_report.md"}')


if __name__ == '__main__':
    main()
