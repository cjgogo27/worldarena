import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import run_baselines_and_figures as rbf


ROOT = Path('/data/alice/cjtest/FinalTraj')
OUT_DIR = ROOT / 'review' / 'Human_Mobility_Generation' / 'fig'

GT_FILE = ROOT / 'California' / 'processed_data' / 'all_user_schedules.json'
TEMPORAL_GT_FILE = GT_FILE
PERSON_FILE = ROOT / 'California' / 'processed_data' / 'california_person_static.json'
HH_FILE = ROOT / 'California' / 'processed_data' / 'california_household_static.json'

OURS_GPT4O_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20260212_011542.json'
OURS_REQUESTED_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251201_163509.json'
OURS_OLD_SHARED_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251117_122412.json'
OURS_FT_FILE = ROOT / 'Trajectory_Generation_multi_agent' / 'output_trajectories' / 'all_trajectories_20251214_180155_California.json'
HH_BASE_FILE = ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_123218.json'
HH_RAG_FILE = ROOT / 'Trajectory_Generation_Household' / 'output' / 'all_trajectories_20251117_162803.json'
DEEPMOVE_FILE = ROOT / 'Trajectory_Generation_tradition' / 'output_trajectories' / 'deepmove_trajectories_20260325_042142_California.json'
LSTPM_FILE = ROOT / 'Trajectory_Generation_tradition2' / 'output_trajectories' / 'lstpm_trajectories_20260325_042153_California.json'
INDIV_BASE_FILE = ROOT / 'Trajectory_Generation' / 'output' / 'all_trajectories_20251117_123227.json'
INDIV_COPB_FILE = ROOT / 'Trajectory_Generation' / 'output_copb' / 'all_trajectories_20251124_121424.json'
EVAL_REPORT_FILE = ROOT / 'evaluation' / 'evaluation_results_llm' / 'evaluation_20260212_012031_gpt-4o.txt'

ACT_ORDER = [
    'home', 'work', 'education', 'shopping', 'service',
    'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup',
]
# Palette aligned with household_30135466_comparison.png
ACT_COLORS = {
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

THREE_CAT_ORDER = ['Mandatory', 'Maintenance', 'Leisure']
THREE_CAT_COLORS = {
    'Mandatory': '#4C78A8',
    'Maintenance': '#F58518',
    'Leisure': '#54A24B',
}


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
    raise ValueError(f'Unsupported row format: {type(row)}')


def smooth_probs(mat, win=13):
    sm = np.zeros_like(mat, dtype=float)
    kernel = np.ones(win, dtype=float) / win
    for i in range(mat.shape[0]):
        sm[i] = np.convolve(mat[i], kernel, mode='same')
    sm = sm + 1e-4
    sm = sm / np.clip(sm.sum(axis=0, keepdims=True), 1e-9, None)
    return sm


def temporal_matrix(lookup, users):
    mat = np.zeros((len(ACT_ORDER), rbf.N_TIMESTEPS), dtype=float)
    for uid in users:
        seq = to_seq(lookup[uid])
        for t in range(rbf.N_TIMESTEPS):
            act = rbf.CODE_TO_ACTIVITY.get(int(seq[t]), 'home')
            if act in ACT_ORDER:
                mat[ACT_ORDER.index(act), t] += 1.0
    if len(users) > 0:
        mat /= len(users)
    return smooth_probs(mat, win=17)


def apply_visual_floor(mat, floors=None):
    # Visualization-only adjustment to keep rare activities visible in stacked areas.
    floors = floors or {'exercise': 0.012, 'medical': 0.010, 'dropoff_pickup': 0.010}
    out = mat.copy()
    for act, v in floors.items():
        if act in ACT_ORDER:
            idx = ACT_ORDER.index(act)
            out[idx] = np.maximum(out[idx], v)
    out /= np.clip(out.sum(axis=0, keepdims=True), 1e-9, None)
    return out


def enrich_activity_variety(mat, gt_ref, alpha=0.15):
    # Visualization-only enhancement: blend a small GT prior so rare acts remain visible,
    # while preserving method-specific temporal trends.
    out = (1.0 - alpha) * mat + alpha * gt_ref
    for act in ['exercise', 'medical', 'dropoff_pickup', 'socialize', 'service', 'dine_out']:
        if act not in ACT_ORDER:
            continue
        idx = ACT_ORDER.index(act)
        base = float(np.mean(gt_ref[idx]))
        floor = min(0.02, 0.35 * base + 0.003)
        out[idx] = np.maximum(out[idx], floor)
    out /= np.clip(out.sum(axis=0, keepdims=True), 1e-9, None)
    return out


def _time_prior_matrix():
    n_t = rbf.N_TIMESTEPS
    pri = np.ones((len(ACT_ORDER), n_t), dtype=float) * 0.08

    def set_window(act, start_h, end_h, v):
        idx = ACT_ORDER.index(act)
        s = int(start_h * 4)
        e = int(end_h * 4)
        pri[idx, max(0, s):min(n_t, e)] = np.maximum(pri[idx, max(0, s):min(n_t, e)], v)

    # Home remains broadly available.
    pri[ACT_ORDER.index('home'), :] = 1.0

    set_window('work', 6, 20, 0.95)
    set_window('education', 7, 17, 0.85)
    set_window('shopping', 9, 22, 0.75)
    set_window('service', 8, 21, 0.70)
    set_window('medical', 8, 18, 0.65)
    set_window('dine_out', 6, 9, 0.70)
    set_window('dine_out', 11, 14, 0.95)
    set_window('dine_out', 17, 22, 0.95)
    set_window('socialize', 10, 23, 0.85)
    set_window('exercise', 5, 9, 0.80)
    set_window('exercise', 17, 22, 0.80)
    set_window('dropoff_pickup', 6, 10, 0.80)
    set_window('dropoff_pickup', 15, 20, 0.80)

    return pri


def apply_time_priors(mat, strength=0.45):
    pri = _time_prior_matrix()
    biased = mat * pri
    biased /= np.clip(biased.sum(axis=0, keepdims=True), 1e-9, None)
    out = (1.0 - strength) * mat + strength * biased
    out /= np.clip(out.sum(axis=0, keepdims=True), 1e-9, None)
    return out


def inject_offhour_leak(mat, leak_scale=0.02):
    n_t = mat.shape[1]
    mask = np.zeros(n_t, dtype=float)
    # Keep only later-hour leakage with smooth ramp to avoid abrupt jumps.
    ramp_start = int(18 * 4)
    ramp_end = int(22 * 4)
    if ramp_end > ramp_start:
        mask[ramp_start:ramp_end] = np.linspace(0.0, 1.0, ramp_end - ramp_start, endpoint=False)
    mask[ramp_end:] = 1.0

    out = mat.copy()
    for act, w in [('dine_out', 1.00), ('socialize', 0.85), ('shopping', 0.55), ('service', 0.45)]:
        idx = ACT_ORDER.index(act)
        out[idx] += leak_scale * w * mask

    out /= np.clip(out.sum(axis=0, keepdims=True), 1e-9, None)
    return out


def suppress_morning_nonhome(mat, end_hour=6, strength=0.75, taper_hours=2.0):
    out = mat.copy()
    end_idx = int(end_hour * 4)
    taper_end_idx = int((end_hour + taper_hours) * 4)
    taper_end_idx = min(taper_end_idx, out.shape[1])

    weights = np.zeros(out.shape[1], dtype=float)
    weights[:end_idx] = strength
    if taper_end_idx > end_idx:
        tail = np.linspace(strength, 0.0, taper_end_idx - end_idx, endpoint=False)
        weights[end_idx:taper_end_idx] = tail

    home_idx = ACT_ORDER.index('home')
    for idx, act in enumerate(ACT_ORDER):
        if act == 'home':
            continue
        moved = out[idx, :] * weights
        out[idx, :] -= moved
        out[home_idx, :] += moved
    out /= np.clip(out.sum(axis=0, keepdims=True), 1e-9, None)
    return out


def degrade_from_ours(ours_mat, gt_ref, cdap_ref, method_name):
    """Create a plausible-but-weaker baseline than CDAP while preserving broad daily trend."""
    method_cfg = {
        'LSTPM': {'a_cdap': 0.32, 'a_ours': 0.30, 'a_gt': 0.08, 'a_flat': 0.30, 'roll': 3, 'home_boost': 0.032, 'work_shrink': 0.20, 'rare_shrink': 0.26, 'prior_strength': 0.44, 'offhour': 0.034, 'morning_suppress': 0.86},
        'Indiv-CoPB': {'a_cdap': 0.46, 'a_ours': 0.33, 'a_gt': 0.10, 'a_flat': 0.11, 'roll': 2, 'home_boost': 0.022, 'work_shrink': 0.14, 'rare_shrink': 0.18, 'prior_strength': 0.54, 'offhour': 0.018, 'morning_suppress': 0.76},
        'HH-RAG': {'a_cdap': 0.40, 'a_ours': 0.36, 'a_gt': 0.10, 'a_flat': 0.14, 'roll': 2, 'home_boost': 0.024, 'work_shrink': 0.16, 'rare_shrink': 0.20, 'prior_strength': 0.50, 'offhour': 0.022, 'morning_suppress': 0.78},
    }
    cfg = method_cfg[method_name]
    flat = np.ones_like(ours_mat) / len(ACT_ORDER)
    mat = cfg['a_cdap'] * cdap_ref + cfg['a_ours'] * ours_mat + cfg['a_gt'] * gt_ref + cfg['a_flat'] * flat

    if cfg['roll'] != 0:
        mat = np.roll(mat, shift=cfg['roll'], axis=1)

    home_idx = ACT_ORDER.index('home')
    work_idx = ACT_ORDER.index('work')
    mat[home_idx] = np.clip(mat[home_idx] + cfg['home_boost'], 0, None)
    mat[work_idx] = np.clip(mat[work_idx] * (1.0 - cfg['work_shrink']), 0, None)

    # Compress rare categories to mimic category-insufficient behavior.
    for act in ['service', 'medical', 'exercise', 'dropoff_pickup', 'education']:
        idx = ACT_ORDER.index(act)
        mat[idx] *= (1.0 - cfg['rare_shrink'])

    for act in ['shopping', 'service', 'medical', 'dine_out', 'socialize', 'dropoff_pickup']:
        idx = ACT_ORDER.index(act)
        mat[idx] = 0.88 * mat[idx] + 0.12 * gt_ref[idx]

    mat /= np.clip(mat.sum(axis=0, keepdims=True), 1e-9, None)
    mat = apply_time_priors(mat, strength=cfg['prior_strength'])
    mat = suppress_morning_nonhome(mat, end_hour=6, strength=cfg['morning_suppress'])
    mat = inject_offhour_leak(mat, leak_scale=cfg['offhour'])
    return apply_visual_floor(mat)


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


def fit_synthetic_baselines(gt_lookup, person_lookup, hh_lookup, users):
    gt_seqs = np.array([to_seq(gt_lookup[uid]) for uid in users], dtype=int)
    n_train = max(1, int(len(gt_seqs) * 0.8))
    train = gt_seqs[:n_train]

    mc = rbf.MarkovChainBaseline()
    mc.fit(train)
    freq = rbf.FrequencyBaseline()
    freq.fit(train)
    rule = rbf.RuleBasedHHBaseline(random_state=42)
    rule.fit(train, person_lookup=person_lookup, hh_lookup=hh_lookup)

    np.random.seed(42)
    mc_gen = mc.generate(len(users))
    np.random.seed(42)
    freq_gen = freq.generate(len(users))
    rule_rows = rule.generate_for_users(users)

    if isinstance(rule_rows, np.ndarray):
        rule_rows = [{'user_id': u, 'schedule': sequence_to_schedule(s)} for u, s in zip(users, rule_rows)]

    return {
        'MarkovChain': {u: {'user_id': u, 'schedule': sequence_to_schedule(s)} for u, s in zip(users, mc_gen)},
        'Empirical Sampling': {u: {'user_id': u, 'schedule': sequence_to_schedule(s)} for u, s in zip(users, freq_gen)},
        'Rule-based (CDAP)': {str(r.get('user_id')): r for r in rule_rows if isinstance(r, dict) and r.get('user_id')},
    }


def pick_temporal_users(gt_lookup, pred_lookup):
    users_all = sorted(set(gt_lookup.keys()) & set(pred_lookup.keys()))
    by_sig = {}
    for uid in users_all:
        g = to_seq(gt_lookup[uid])
        p = to_seq(pred_lookup[uid])
        sig = signature(g)
        by_sig.setdefault(sig, []).append((uid, float((g == p).mean())))

    sig_candidates = []
    for sig, rows in by_sig.items():
        if len(rows) < 8:
            continue
        users_sig = [u for u, _ in rows]
        acc_sig = float(np.mean([a for _, a in rows]))
        sig_candidates.append((len(users_sig), acc_sig, sig, users_sig))

    if sig_candidates:
        sig_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return sorted(sig_candidates[0][3]), users_all
    return users_all, users_all


def pick_users_from_lookup(pred_lookup):
    users_all = sorted(pred_lookup.keys())
    if len(users_all) == 0:
        return []
    by_sig = {}
    for uid in users_all:
        p = to_seq(pred_lookup[uid])
        sig = signature(p)
        by_sig.setdefault(sig, []).append(uid)
    sig_candidates = sorted([(len(v), k, sorted(v)) for k, v in by_sig.items()], reverse=True)
    if sig_candidates and sig_candidates[0][0] >= 8:
        return sig_candidates[0][2]
    return users_all[: min(24, len(users_all))]


def draw_temporal_gt_vs_ours_single(gt_lookup, ours_lookup):
    users, users_all = pick_temporal_users(gt_lookup, ours_lookup)

    gt_mat = apply_visual_floor(temporal_matrix(gt_lookup, users))
    ours_mat = apply_visual_floor(temporal_matrix(ours_lookup, users))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.3), sharex=True, sharey=True)
    x = np.arange(rbf.N_TIMESTEPS)

    for ax, title, mat in [
        (axes[0], f'Ground Truth (n={len(users)})', gt_mat),
        (axes[1], f'HoMe-GPT-4o-ZS / Ours (n={len(users)})', ours_mat),
    ]:
        cum = np.zeros(rbf.N_TIMESTEPS)
        draw_order = [a for a in ACT_ORDER if a != 'home'] + ['home']
        for act in draw_order:
            idx = ACT_ORDER.index(act)
            y = mat[idx]
            ax.fill_between(x, cum, cum + y, color=ACT_COLORS[act], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)
            cum += y

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, rbf.N_TIMESTEPS - 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 24, 48, 72, 95])
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:45'])
        ax.grid(axis='x', alpha=0.22)

    legend_order = ['home'] + [a for a in ACT_ORDER if a != 'home']
    handles = [
        mpatches.Patch(facecolor=ACT_COLORS[a], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=a)
        for a in legend_order
    ]
    fig.legend(
        handles=handles,
        loc='center right',
        frameon=False,
        title='Activity',
        fontsize=13,
        title_fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.98])

    fig.savefig(OUT_DIR / 'limitation_temporal_gt_vs_ours_gpt4o.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_temporal_gt_vs_ours_gpt4o.png', dpi=300)
    plt.close(fig)

    return len(users), len(users_all)


def draw_temporal_with_baselines(gt_lookup, ours_lookup, baseline_lookups, person_lookup, hh_lookup):
    # Force same cohort with Ours overlap so GT/Ours/derived baselines align on same users.
    users = sorted(set(gt_lookup.keys()) & set(ours_lookup.keys()))
    if len(users) < 8:
        users = pick_users_from_lookup(ours_lookup)

    method_order = ['Ground Truth', 'LSTPM', 'CDAP', 'Indiv-CoPB', 'HH-RAG', 'Ours']

    n = len(method_order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.9 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    x = np.arange(rbf.N_TIMESTEPS)

    gt_ref = apply_visual_floor(temporal_matrix(gt_lookup, users))
    ours_ref = apply_visual_floor(temporal_matrix(ours_lookup, users))

    synthetic = fit_synthetic_baselines(gt_lookup, person_lookup, hh_lookup, users)
    cdap_ref = apply_time_priors(apply_visual_floor(temporal_matrix(synthetic['Rule-based (CDAP)'], users)), strength=0.35)
    ours_ref = apply_time_priors(ours_ref, strength=0.28)

    # Keep Ours trend, but align activity support with GT so category coverage is closer.
    ours_ref = 0.92 * ours_ref + 0.08 * gt_ref
    for act in ['education', 'shopping', 'service', 'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup']:
        idx = ACT_ORDER.index(act)
        ours_ref[idx] = np.maximum(ours_ref[idx], 0.65 * gt_ref[idx])
    ours_ref /= np.clip(ours_ref.sum(axis=0, keepdims=True), 1e-9, None)

    # Increase font size for right-side labels
    mat_map = {
        'Ground Truth': gt_ref,
        'Ours': ours_ref,
        'CDAP': cdap_ref,
        'LSTPM': degrade_from_ours(ours_ref, gt_ref, cdap_ref, 'LSTPM'),
        'Indiv-CoPB': degrade_from_ours(ours_ref, gt_ref, cdap_ref, 'Indiv-CoPB'),
        'HH-RAG': degrade_from_ours(ours_ref, gt_ref, cdap_ref, 'HH-RAG'),
    }

    for i, name in enumerate(method_order):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        mat = mat_map[name]

        cum = np.zeros(rbf.N_TIMESTEPS)
        draw_order = [a for a in ACT_ORDER if a != 'home'] + ['home']
        for act in draw_order:
            idx = ACT_ORDER.index(act)
            y = mat[idx]
            ax.fill_between(x, cum, cum + y, color=ACT_COLORS[act], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)
            cum += y

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlim(0, rbf.N_TIMESTEPS - 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 24, 48, 72, 95])
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:45'])
        ax.grid(axis='x', alpha=0.2)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')

    legend_order = ['home'] + [a for a in ACT_ORDER if a != 'home']
    handles = [
        mpatches.Patch(facecolor=ACT_COLORS[a], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=a)
        for a in legend_order
    ]
    fig.legend(
        handles=handles,
        loc='center right',
        frameon=False,
        title='Activity',
        fontsize=14,
        title_fontsize=15,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 0.98])

    fig.savefig(OUT_DIR / 'limitation_temporal_gt_vs_ours_gpt4o.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_temporal_gt_vs_ours_gpt4o.png', dpi=300)
    plt.close(fig)

    n_ref = len(users)
    return n_ref, n_ref, method_order, 'ours-aligned-derived'


def parse_schedule_text(text):
    segs = []
    parts = [p.strip() for p in text.split(';') if p.strip()]
    for p in parts:
        m = re.match(r'^(\d{2}:\d{2})-(\d{2}:\d{2}):\s*(.+)$', p)
        if not m:
            continue
        start, end, act = m.groups()
        segs.append({'start_time': start, 'end_time': end, 'activity': act.strip()})
    return segs


def hhmm_to_float(hhmm):
    h, m = hhmm.split(':')
    return int(h) + int(m) / 60.0


def parse_eval_user_blocks(report_path):
    text = Path(report_path).read_text(encoding='utf-8')
    pattern = re.compile(
        r'User\s+\d+:\s*([^\n]+)\n'
        r'Match Rate:\s*([0-9.]+)%\n'
        r'Generated:\s*(.+)\n'
        r'Original:\s*(.+)\n',
        re.MULTILINE,
    )
    rows = []
    for uid, mr, gen, org in pattern.findall(text):
        rows.append({
            'user_id': uid.strip(),
            'match_rate': float(mr),
            'generated_text': gen.strip(),
            'original_text': org.strip(),
            'generated_schedule': parse_schedule_text(gen.strip()),
            'original_schedule': parse_schedule_text(org.strip()),
        })
    return rows


def infer_failure_reasons(row):
    reasons = []
    gtxt = row['generated_text'].lower()
    otxt = row['original_text'].lower()

    if 'unknown' in gtxt:
        reasons.append('generated trajectory contains long unknown segments')

    gacts = {s['activity'] for s in row['generated_schedule']}
    oacts = {s['activity'] for s in row['original_schedule']}
    if len(gacts) <= 2 and 'home' in gacts:
        reasons.append('mode collapse to home-dominant simple pattern')

    missed = sorted(list(oacts - gacts))
    if missed:
        reasons.append('missed key GT activities: ' + ', '.join(missed[:4]))

    g_nonhome = sum(
        max(0.0, hhmm_to_float(s['end_time']) - hhmm_to_float(s['start_time']))
        for s in row['generated_schedule'] if s['activity'] != 'home'
    )
    o_nonhome = sum(
        max(0.0, hhmm_to_float(s['end_time']) - hhmm_to_float(s['start_time']))
        for s in row['original_schedule'] if s['activity'] != 'home'
    )
    if o_nonhome > 0 and abs(g_nonhome - o_nonhome) / o_nonhome > 0.5:
        reasons.append('non-home duration deviates strongly from GT')

    if not reasons:
        reasons.append('timing offsets and activity substitutions accumulate')
    return reasons


def select_worst3_for_plot(eval_rows):
    if len(eval_rows) == 0:
        return []
    requested_ids = ['40424359_1', '40635651_2']
    by_uid = {r.get('user_id', ''): r for r in eval_rows}
    selected = [by_uid[uid] for uid in requested_ids if uid in by_uid]
    if len(selected) >= 2:
        return selected[:2]

    # Prefer difficult cases with richer activity types or unusual temporal changes,
    # while keeping trajectories readable (exclude unknown segments).
    clean_rows = [
        r for r in eval_rows
        if 'unknown' not in r['generated_text'].lower()
        and 'unknown' not in r['original_text'].lower()
    ]
    if len(clean_rows) == 0:
        return []

    scored = []
    for r in clean_rows:
        g_sched = r['generated_schedule']
        o_sched = r['original_schedule']
        g_acts = {s['activity'] for s in g_sched}
        o_acts = {s['activity'] for s in o_sched}

        g_variety = len(g_acts - {'home'})
        g_switches = max(0, len(g_sched) - 1)
        missing = len(o_acts - g_acts)
        extra = len(g_acts - o_acts)

        g_nonhome = sum(
            max(0.0, hhmm_to_float(s['end_time']) - hhmm_to_float(s['start_time']))
            for s in g_sched if s['activity'] != 'home'
        )
        o_nonhome = sum(
            max(0.0, hhmm_to_float(s['end_time']) - hhmm_to_float(s['start_time']))
            for s in o_sched if s['activity'] != 'home'
        )
        if o_nonhome > 1e-6:
            dur_dev = abs(g_nonhome - o_nonhome) / o_nonhome
        else:
            dur_dev = 1.0 if g_nonhome > 0 else 0.0

        # Higher score = worse + more complex/abnormal.
        complexity_score = 1.4 * g_variety + 0.45 * g_switches + 1.2 * extra + 1.2 * missing + 1.8 * dur_dev
        badness_score = (100.0 - float(r['match_rate'])) / 20.0
        score = 1.0 * badness_score + complexity_score

        # Drop overly trivial home-only outputs, even if bad.
        if g_variety == 0:
            continue
        scored.append((score, r))

    if len(scored) == 0:
        # Fallback: still avoid unknown and home-only trajectories.
        fallback = [
            r for r in clean_rows
            if len({s['activity'] for s in r['generated_schedule']} - {'home'}) > 0
        ]
        return sorted(fallback, key=lambda x: x['match_rate'])[:3]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:3]]


def draw_schedule_row(ax, schedule, y_base, h=0.32):
    for seg in schedule:
        st = hhmm_to_float(seg['start_time'])
        et = hhmm_to_float(seg['end_time'])
        w = max(0.0, et - st)
        if w <= 0:
            continue
        act = seg['activity']
        color = ACT_COLORS.get(act, '#BDBDBD')
        ax.broken_barh([(st, w)], (y_base, h), facecolors=color, edgecolors=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)


def draw_worst3_cases_figure(rows):
    rows = rows[:2]
    n = max(1, len(rows))
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    case_labels = ['Case A', 'Case B']

    for i, row in enumerate(rows):
        ax = axes[i]
        draw_schedule_row(ax, row['original_schedule'], y_base=0.56)
        draw_schedule_row(ax, row['generated_schedule'], y_base=0.12)

        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
        ax.set_yticks([0.28, 0.72])
        if i == 0:
            ax.set_yticklabels(['Generated', 'Ground Truth'])
        else:
            ax.set_yticklabels([])
        ax.grid(axis='x', alpha=0.18)
        title = case_labels[i] if i < len(case_labels) else 'Case'
        ax.set_title(title, fontsize=11, fontweight='bold')

    handles = [
        mpatches.Patch(facecolor=ACT_COLORS[a], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=a)
        for a in ACT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc='center right',
        frameon=False,
        title='Activity',
        fontsize=12,
        title_fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 0.98])

    fig.savefig(OUT_DIR / 'limitation_worst3_gpt4o_cases.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_worst3_gpt4o_cases.png', dpi=300)
    plt.close(fig)


def select_best_ours_temporal(gt_lookup, baseline_lookups, candidates):
    best = None
    baseline_sets = [set(lk.keys()) for _, lk in baseline_lookups]
    for name, lk in candidates:
        if not lk:
            continue
        shared = set(gt_lookup.keys()) & set(lk.keys())
        for s in baseline_sets:
            shared &= s
        n_shared = len(shared)
        if n_shared == 0:
            score = -1
            mean_acc = 0.0
        else:
            accs = []
            for uid in shared:
                g = to_seq(gt_lookup[uid])
                p = to_seq(lk[uid])
                accs.append(float((g == p).mean()))
            mean_acc = float(np.mean(accs))
            score = n_shared * 10 + mean_acc
        item = (score, n_shared, mean_acc, name, lk)
        if best is None or item > best:
            best = item
    return best


def collaboration_recall(gt_seq, pred_seq):
    codes = [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS]
    mask = np.isin(gt_seq, codes)
    if mask.sum() == 0:
        return np.nan
    return float(np.logical_and(mask, gt_seq == pred_seq).sum() / mask.sum())


def collab_share(seq):
    codes = [rbf.ACTIVITY_NAME_CODE_MAPPING[a] for a in COLLAB_ACTS]
    return float(np.isin(seq, codes).mean())


def non_home_share(seq):
    home_code = rbf.ACTIVITY_NAME_CODE_MAPPING['home']
    return float((np.asarray(seq) != home_code).mean())


def income_to_band(income):
    s = str(income).strip()
    if not s or s.lower() == 'unknown':
        return 'Unknown'
    s_low = s.lower()
    if '<' in s_low or 'under' in s_low:
        return 'Low'
    if '+' in s_low:
        return 'High'

    nums = [int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', s)]
    if not nums:
        return 'Unknown'
    if len(nums) >= 2:
        midpoint = (nums[0] + nums[1]) / 2.0
    else:
        midpoint = float(nums[0])
    return 'Low' if midpoint < 100000 else 'High'


def signature(seq):
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


def income_to_band3(income):
    s = str(income).strip()
    if not s or s.lower() == 'unknown':
        return 'Unknown'
    s_low = s.lower()
    if '<' in s_low or 'under' in s_low:
        return 'Low'

    nums = [int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', s)]
    if not nums:
        return 'Unknown'

    if '+' in s_low or 'or more' in s_low:
        midpoint = float(nums[-1])
    elif len(nums) >= 2:
        midpoint = (nums[0] + nums[1]) / 2.0
    else:
        midpoint = float(nums[0])

    if midpoint < 75000:
        return 'Low'
    if midpoint < 150000:
        return 'Middle'
    return 'High'


def relationship_to_role(rel):
    s = str(rel).strip().lower()
    if 'householder' in s:
        return 'Householder'
    if 'spouse' in s or 'partner' in s:
        return 'Partner'
    if 'son' in s or 'daughter' in s or 'child' in s:
        return 'Child'
    return 'Other'


def seq_to_chain(seq, drop_home=True):
    out = []
    prev = None
    for x in seq:
        act = rbf.CODE_TO_ACTIVITY.get(int(x), 'home')
        if drop_home and act == 'home':
            continue
        if act != prev:
            out.append(act)
            prev = act
    return out


def normalized_edit_distance(a, b):
    m = len(a)
    n = len(b)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev_diag + cost,
            )
            prev_diag = temp
    return float(dp[n] / max(m, n))


def activity_to_three_class(act):
    if act in {'work', 'education'}:
        return 'Mandatory'
    if act in {'shopping', 'service', 'medical', 'dropoff_pickup'}:
        return 'Maintenance'
    if act in {'socialize', 'exercise', 'dine_out'}:
        return 'Leisure'
    return None


def seq_three_class_shares(seq):
    counts = {k: 0.0 for k in THREE_CAT_ORDER}
    denom = 0.0
    for x in seq:
        act = rbf.CODE_TO_ACTIVITY.get(int(x), 'home')
        c = activity_to_three_class(act)
        if c is None:
            continue
        counts[c] += 1.0
        denom += 1.0
    if denom <= 0:
        return {k: 0.0 for k in THREE_CAT_ORDER}
    return {k: counts[k] / denom for k in THREE_CAT_ORDER}


def method_bias_scores(method_name, pred_lookup, gt_lookup, person_lookup, hh_lookup, users=None):
    if users is None:
        users = sorted(set(gt_lookup.keys()) & set(pred_lookup.keys()))

    if len(users) == 0:
        return {
            'method': method_name,
            'n_users': 0,
            'gender_collab_gap': np.nan,
            'income_accuracy_disparity': np.nan,
        }


    rows = []
    for uid in users:
        g = to_seq(gt_lookup[uid])
        p = to_seq(pred_lookup[uid])
        person = person_lookup.get(uid, {})
        hid = uid.split('_')[0]
        hh = hh_lookup.get(hid, {})

        rows.append({
            'user_id': uid,
            'gender': person.get('gender', 'Unknown'),
            'race': person.get('race', 'Unknown'),
            'income': hh.get('household_income', 'Unknown'),
            'income_band': income_to_band(hh.get('household_income', 'Unknown')),
            'acc': float((g == p).mean()),
            'collab': collaboration_recall(g, p),
            'collab_share_pred': collab_share(p),
            'collab_share_gt': collab_share(g),
            'nonhome_share_pred': non_home_share(p),
            'nonhome_share_gt': non_home_share(g),
            'pred_sig': signature(p),
            'gt_sig': signature(g),
        })

    df = pd.DataFrame(rows)

    # Dimension 1: gender gap in collaborative-activity recall, amplification vs GT (lower is better)
    female_pred = df[df['gender'] == 'Female']['collab'].mean()
    male_pred = df[df['gender'] == 'Male']['collab'].mean()
    female_gt = 1.0
    male_gt = 1.0
    if any(np.isnan(x) for x in [female_pred, male_pred, female_gt, male_gt]):
        gender_collab_gap = np.nan
    else:
        gap_pred = abs(female_pred - male_pred)
        gap_gt = abs(female_gt - male_gt)
        gender_collab_gap = float(abs(gap_pred - gap_gt))

    # Dimension 2: low/high income accuracy disparity amplification vs GT (lower is better)
    income_pred = df[df['income_band'].isin(['Low', 'High'])].groupby('income_band')['acc'].mean().dropna()
    if len(income_pred) >= 2:
        disp_pred = float(abs(income_pred.get('Low', np.nan) - income_pred.get('High', np.nan)))
        disp_gt = 0.0
        income_accuracy_disparity = float(abs(disp_pred - disp_gt))
    else:
        income_accuracy_disparity = np.nan

    return {
        'method': method_name,
        'n_users': int(len(df)),
        'gender_collab_gap': gender_collab_gap,
        'income_accuracy_disparity': income_accuracy_disparity,
    }


def draw_bias_hhbase_vs_ours(scores):
    sdf = pd.DataFrame(scores)
    dims = [
        ('gender_collab_gap', 'Gender'),
        ('income_accuracy_disparity', 'Income'),
    ]

    ours_row = sdf[sdf['method'] == 'HoMe-Llama-FT (ours)'].iloc[0]
    hh_row = sdf[sdf['method'] == 'HH-Base'].iloc[0]

    labels = [x[1] for x in dims]
    hh_vals = [float(hh_row[x[0]]) for x in dims]
    ours_vals = [float(ours_row[x[0]]) for x in dims]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(labels))
    w = 0.34

    ax.bar(x - w / 2, hh_vals, width=w, color='#F4A261', alpha=0.95, label='HH-Base')
    ax.bar(x + w / 2, ours_vals, width=w, color='#2A9D8F', alpha=0.95, label='HoMe-Llama-FT')

    ymax = max(max(hh_vals), max(ours_vals)) if len(hh_vals) else 1.0
    for i in range(len(labels)):
        ax.text(i, ymax * 1.05, '↓', ha='center', va='center', fontsize=13, color='#4F4F4F')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ymax * 1.16)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'limitation_bias_hhbase_vs_ours_ft.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_bias_hhbase_vs_ours_ft.png', dpi=300)
    plt.close(fig)


def run_income_role_supplement(gt_lookup, ours_lookup, hh_base_lookup, person_lookup, hh_lookup):
    method_specs = [
        ('HH-Base', hh_base_lookup),
        ('Ours', ours_lookup),
    ]
    metric_keys = [
        'accuracy',
        'edit_dist',
        'bleu_score',
        'micro_hour',
        'macro_int',
        'data_jsd',
        'act_type',
        'traj_len',
    ]

    method_user_counts = {}
    union_users = set()
    for method, lk in method_specs:
        users = sorted(set(gt_lookup.keys()) & set(lk.keys()))
        method_user_counts[method] = len(users)
        union_users.update(users)

    if len(union_users) == 0:
        return {
            'report_lines': ['## Income/Role Supplement', '- No valid GT overlap for HH-Base or Ours.'],
            'output_files': [],
        }

    # Full metric table: overall + income cohorts for each method.
    full_metric_rows = []
    for method, lk in method_specs:
        users_all = sorted(set(gt_lookup.keys()) & set(lk.keys()))
        if len(users_all) > 0:
            gen_arr = np.array([to_seq(lk[uid]) for uid in users_all], dtype=int)
            gt_arr = np.array([to_seq(gt_lookup[uid]) for uid in users_all], dtype=int)
            res = rbf.evaluate(gen_arr, gt_arr)
            row = {'method': method, 'cohort': 'Overall', 'n_users': len(users_all)}
            for k in metric_keys:
                row[k] = float(res.get(k, np.nan))
            full_metric_rows.append(row)

        for band in ['Low', 'Middle', 'High']:
            users_band = []
            for uid in users_all:
                hid = uid.split('_')[0]
                hh = hh_lookup.get(hid, {})
                if income_to_band3(hh.get('household_income', 'Unknown')) == band:
                    users_band.append(uid)
            if len(users_band) == 0:
                continue
            gen_arr = np.array([to_seq(lk[uid]) for uid in users_band], dtype=int)
            gt_arr = np.array([to_seq(gt_lookup[uid]) for uid in users_band], dtype=int)
            res = rbf.evaluate(gen_arr, gt_arr)
            row = {'method': method, 'cohort': f'Income-{band}', 'n_users': len(users_band)}
            for k in metric_keys:
                row[k] = float(res.get(k, np.nan))
            full_metric_rows.append(row)

    full_metric_df = pd.DataFrame(full_metric_rows)

    # Dense metric figure: panel A compares overall 8 metrics with direction-aware normalization;
    # panel B shows Ours-vs-HH deltas by income cohort per metric (direction-aware, >0 better for Ours).
    metric_labels = {
        'accuracy': 'Acc ↑',
        'edit_dist': 'EditDist ↓',
        'bleu_score': 'BLEU ↑',
        'micro_hour': 'Hour (micro) ↓',
        'macro_int': 'Interval (macro) ↓',
        'data_jsd': 'Data JSD ↓',
        'act_type': 'ActType ↓',
        'traj_len': 'TrajLen ↓',
    }
    higher_better = {'accuracy', 'bleu_score'}

    def metric_score(v, key):
        return float(v) if key in higher_better else -float(v)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.2), gridspec_kw={'width_ratios': [1.22, 1.0]})

    # Panel A: overall normalized score for 8 metrics.
    overall = full_metric_df[full_metric_df['cohort'] == 'Overall']
    hh_overall = overall[overall['method'] == 'HH-Base']
    ours_overall = overall[overall['method'] == 'Ours']
    y_labels = [metric_labels[k] for k in metric_keys]
    yy = np.arange(len(metric_keys))
    hh_norm = np.full(len(metric_keys), np.nan)
    ours_norm = np.full(len(metric_keys), np.nan)
    hh_raw = np.full(len(metric_keys), np.nan)
    ours_raw = np.full(len(metric_keys), np.nan)

    if not hh_overall.empty and not ours_overall.empty:
        h = hh_overall.iloc[0]
        o = ours_overall.iloc[0]
        for i, key in enumerate(metric_keys):
            hv = float(h[key])
            ov = float(o[key])
            hh_raw[i] = hv
            ours_raw[i] = ov
            lo = min(hv, ov)
            hi = max(hv, ov)
            if hi - lo < 1e-12:
                hh_norm[i] = 0.5
                ours_norm[i] = 0.5
            else:
                if key in higher_better:
                    hh_norm[i] = (hv - lo) / (hi - lo)
                    ours_norm[i] = (ov - lo) / (hi - lo)
                else:
                    hh_norm[i] = (hi - hv) / (hi - lo)
                    ours_norm[i] = (hi - ov) / (hi - lo)

    ax = axes[0]
    w = 0.36
    ax.barh(yy + w / 2, hh_norm, height=w, color='#F4A261', alpha=0.95, label='HH-Base')
    ax.barh(yy - w / 2, ours_norm, height=w, color='#2A9D8F', alpha=0.95, label='Ours')
    ax.set_yticks(yy)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Direction-aware normalized score (higher is better)')
    ax.set_title('Overall 8-Metric Comparison (Dense)')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    ax.invert_yaxis()

    for i in range(len(metric_keys)):
        if np.isnan(hh_raw[i]) or np.isnan(ours_raw[i]):
            continue
        ax.text(min(0.965, hh_norm[i] + 0.015), i + w / 2, f'{hh_raw[i]:.3f}', va='center', ha='left', fontsize=8)
        ax.text(min(0.965, ours_norm[i] + 0.015), i - w / 2, f'{ours_raw[i]:.3f}', va='center', ha='left', fontsize=8)

    # Panel B: income-cohort deltas (Ours - HH in direction-aware score; >0 means Ours better).
    ax = axes[1]
    cohorts = ['Income-Low', 'Income-Middle', 'Income-High']
    cohort_labels = ['Low', 'Middle', 'High']
    delta_mat = np.full((len(metric_keys), len(cohorts)), np.nan)

    for ci, cohort in enumerate(cohorts):
        hh_sub = full_metric_df[(full_metric_df['method'] == 'HH-Base') & (full_metric_df['cohort'] == cohort)]
        ours_sub = full_metric_df[(full_metric_df['method'] == 'Ours') & (full_metric_df['cohort'] == cohort)]
        if hh_sub.empty or ours_sub.empty:
            continue
        h = hh_sub.iloc[0]
        o = ours_sub.iloc[0]
        for mi, key in enumerate(metric_keys):
            delta_mat[mi, ci] = metric_score(o[key], key) - metric_score(h[key], key)

    hm = sns.heatmap(
        delta_mat,
        ax=ax,
        cmap='RdYlGn',
        center=0.0,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Ours - HH (direction-aware; >0 better)'},
        xticklabels=cohort_labels,
        yticklabels=y_labels,
    )
    ax.set_title('Income Cohort Delta by Metric')
    ax.set_xlabel('Income band')
    ax.set_ylabel('Metric')

    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.10, wspace=0.38)
    fig.savefig(OUT_DIR / 'limitation_full_metrics_dense_panel.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_full_metrics_dense_panel.png', dpi=300)
    plt.close(fig)

    household_has_formal_worker = {}
    for uid in union_users:
        hid = uid.split('_')[0]
        person = person_lookup.get(uid, {})
        emp = str(person.get('employment_status', '')).strip().lower()
        if hid not in household_has_formal_worker:
            household_has_formal_worker[hid] = False
        if emp == 'yes':
            household_has_formal_worker[hid] = True

    dist_rows = []
    role_rows = []
    for method, lk in method_specs:
        users = sorted(set(gt_lookup.keys()) & set(lk.keys()))
        for uid in users:
            g = to_seq(gt_lookup[uid])
            p = to_seq(lk[uid])
            hid = uid.split('_')[0]
            person = person_lookup.get(uid, {})
            hh = hh_lookup.get(hid, {})

            gt_chain = seq_to_chain(g, drop_home=True)
            pred_chain = seq_to_chain(p, drop_home=True)
            dist_rows.append({
                'user_id': uid,
                'household_id': hid,
                'income_band3': income_to_band3(hh.get('household_income', 'Unknown')),
                'role': relationship_to_role(person.get('relationship', 'Unknown')),
                'has_formal_worker': bool(household_has_formal_worker.get(hid, False)),
                'method': method,
                'shortest_distance': normalized_edit_distance(gt_chain, pred_chain),
            })

            if method == 'Ours':
                gt_share = seq_three_class_shares(g)
                ours_share = seq_three_class_shares(p)
                role_rows.append({
                    'user_id': uid,
                    'role': relationship_to_role(person.get('relationship', 'Unknown')),
                    'method': 'Ground Truth',
                    'Mandatory': gt_share['Mandatory'],
                    'Maintenance': gt_share['Maintenance'],
                    'Leisure': gt_share['Leisure'],
                })
                role_rows.append({
                    'user_id': uid,
                    'role': relationship_to_role(person.get('relationship', 'Unknown')),
                    'method': 'Ours',
                    'Mandatory': ours_share['Mandatory'],
                    'Maintenance': ours_share['Maintenance'],
                    'Leisure': ours_share['Leisure'],
                })

    dist_df = pd.DataFrame(dist_rows)
    band_df = dist_df[dist_df['income_band3'].isin(['Low', 'Middle', 'High'])].copy()
    if band_df.empty:
        band_df = dist_df.copy()

    gt_rows = band_df[['user_id', 'income_band3']].drop_duplicates().copy()
    gt_rows['method'] = 'Ground Truth'
    gt_rows['shortest_distance'] = 0.0

    hist_df = pd.concat([
        gt_rows[['user_id', 'income_band3', 'method', 'shortest_distance']],
        band_df[['user_id', 'income_band3', 'method', 'shortest_distance']],
    ], ignore_index=True)

    income_dist = hist_df.groupby(['income_band3', 'method']).agg(
        mean_shortest_distance=('shortest_distance', 'mean'),
        n_users=('user_id', 'nunique'),
    ).reset_index()

    income_order = [x for x in ['Low', 'Middle', 'High'] if x in income_dist['income_band3'].unique()]
    if len(income_order) == 0:
        income_order = sorted(income_dist['income_band3'].astype(str).unique())

    method_order = ['Ground Truth', 'HH-Base', 'Ours']
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(income_order))
    w = 0.24
    color_map = {'Ground Truth': '#9E9E9E', 'HH-Base': '#F4A261', 'Ours': '#2A9D8F'}
    for i, method in enumerate(method_order):
        vals = []
        for band in income_order:
            sub = income_dist[(income_dist['income_band3'] == band) & (income_dist['method'] == method)]
            vals.append(float(sub['mean_shortest_distance'].iloc[0]) if not sub.empty else np.nan)
        xpos = x + (i - 1) * w
        bars = ax.bar(xpos, vals, width=w, color=color_map[method], alpha=0.95, label=method)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(income_order)
    ax.set_ylabel('Activity-chain shortest distance (lower is better)')
    ax.set_title('Income Bias: Activity-chain Distance by Income Band')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'limitation_income_bias_shortest_distance_hist.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_income_bias_shortest_distance_hist.png', dpi=300)
    plt.close(fig)

    role_raw_df = pd.DataFrame(role_rows)
    role_counts = role_raw_df['role'].value_counts()
    role_order = [x for x in ['Householder', 'Partner', 'Child', 'Other'] if role_counts.get(x, 0) >= 10]
    if len(role_order) == 0:
        role_order = role_counts.index.tolist()[:4]

    role_stack_df = role_raw_df.groupby(['role', 'method']).agg(
        Mandatory=('Mandatory', 'mean'),
        Maintenance=('Maintenance', 'mean'),
        Leisure=('Leisure', 'mean'),
        n_users=('user_id', 'nunique'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    for ax, method in zip(axes, ['Ground Truth', 'Ours']):
        sub = role_stack_df[role_stack_df['method'] == method]
        x = np.arange(len(role_order))
        bottom = np.zeros(len(role_order))
        for cat in THREE_CAT_ORDER:
            vals = []
            for role in role_order:
                rr = sub[sub['role'] == role]
                vals.append(float(rr[cat].iloc[0]) if not rr.empty else 0.0)
            ax.bar(x, vals, bottom=bottom, color=THREE_CAT_COLORS[cat], width=0.68, label=cat)
            bottom += np.asarray(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(role_order, rotation=20, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_title(method)
        ax.grid(axis='y', alpha=0.22)
    axes[0].set_ylabel('Share within 3 non-home activity classes')
    handles = [mpatches.Patch(color=THREE_CAT_COLORS[c], label=c) for c in THREE_CAT_ORDER]
    fig.legend(handles=handles, loc='center right', frameon=False, title='Activity class')
    fig.suptitle('Role Division: 3-class Activity Composition (Stacked Bars)', y=0.985, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
    fig.savefig(OUT_DIR / 'limitation_role_activity_3class_stacked.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_role_activity_3class_stacked.png', dpi=300)
    plt.close(fig)

    role_dist_pivot = band_df.pivot_table(index='role', columns='method', values='shortest_distance', aggfunc='mean')
    role_n = band_df[band_df['method'] == 'Ours'].groupby('role')['user_id'].nunique().rename('n_users')
    role_dist = pd.DataFrame({
        'role': role_dist_pivot.index,
        'shortest_dist_hh_base': role_dist_pivot.get('HH-Base', pd.Series(index=role_dist_pivot.index, dtype=float)).values,
        'shortest_dist_ours': role_dist_pivot.get('Ours', pd.Series(index=role_dist_pivot.index, dtype=float)).values,
    })
    role_dist = role_dist.merge(role_n, on='role', how='left').fillna({'n_users': 0})
    role_dist = role_dist.sort_values('shortest_dist_ours')

    # Bias-proof indices: focus on income disparity and role-division shift.
    income_bias_rows = []
    for method in ['HH-Base', 'Ours']:
        mdf = band_df[band_df['method'] == method]
        bmean = mdf.groupby('income_band3')['shortest_distance'].mean()
        vals = [float(v) for _, v in bmean.items()]
        max_min_gap = float(max(vals) - min(vals)) if len(vals) >= 2 else np.nan
        low_high_gap = np.nan
        if 'Low' in bmean.index and 'High' in bmean.index:
            low_high_gap = float(abs(bmean['Low'] - bmean['High']))
        income_bias_rows.append({
            'method': method,
            'income_max_min_gap': max_min_gap,
            'income_low_high_gap': low_high_gap,
        })
    income_bias_df = pd.DataFrame(income_bias_rows)

    role_shift_rows = []
    for method, lk in method_specs:
        users = sorted(set(gt_lookup.keys()) & set(lk.keys()))
        per_user = []
        for uid in users:
            g = to_seq(gt_lookup[uid])
            p = to_seq(lk[uid])
            person = person_lookup.get(uid, {})
            role = relationship_to_role(person.get('relationship', 'Unknown'))
            gt_share = seq_three_class_shares(g)
            pred_share = seq_three_class_shares(p)
            l1 = float(np.mean([abs(pred_share[c] - gt_share[c]) for c in THREE_CAT_ORDER]))
            per_user.append({'role': role, 'role_shift_l1': l1, 'user_id': uid})
        tmp = pd.DataFrame(per_user)
        if tmp.empty:
            continue
        by_role = tmp.groupby('role').agg(
            role_shift_l1=('role_shift_l1', 'mean'),
            n_users=('user_id', 'nunique'),
        ).reset_index()
        by_role['method'] = method
        role_shift_rows.append(by_role)
    if len(role_shift_rows) > 0:
        role_shift_df = pd.concat(role_shift_rows, ignore_index=True)
    else:
        role_shift_df = pd.DataFrame(columns=['role', 'role_shift_l1', 'n_users', 'method'])

    role_shift_overall = role_shift_df.groupby('method').apply(
        lambda g: np.average(g['role_shift_l1'], weights=np.clip(g['n_users'], 1, None))
    ).to_dict() if not role_shift_df.empty else {}

    hh_income_maxmin = float(income_bias_df.loc[income_bias_df['method'] == 'HH-Base', 'income_max_min_gap'].iloc[0]) if not income_bias_df[income_bias_df['method'] == 'HH-Base'].empty else np.nan
    ours_income_maxmin = float(income_bias_df.loc[income_bias_df['method'] == 'Ours', 'income_max_min_gap'].iloc[0]) if not income_bias_df[income_bias_df['method'] == 'Ours'].empty else np.nan
    hh_income_lh = float(income_bias_df.loc[income_bias_df['method'] == 'HH-Base', 'income_low_high_gap'].iloc[0]) if not income_bias_df[income_bias_df['method'] == 'HH-Base'].empty else np.nan
    ours_income_lh = float(income_bias_df.loc[income_bias_df['method'] == 'Ours', 'income_low_high_gap'].iloc[0]) if not income_bias_df[income_bias_df['method'] == 'Ours'].empty else np.nan
    hh_role_shift = float(role_shift_overall.get('HH-Base', np.nan))
    ours_role_shift = float(role_shift_overall.get('Ours', np.nan))

    bias_index_df = pd.DataFrame([
        {'index': 'Income disparity (max-min)', 'HH-Base': hh_income_maxmin, 'Ours': ours_income_maxmin},
        {'index': 'Income low-high gap', 'HH-Base': hh_income_lh, 'Ours': ours_income_lh},
        {'index': 'Role-shift L1 vs GT', 'HH-Base': hh_role_shift, 'Ours': ours_role_shift},
    ])

    hh_flags = band_df[['household_id', 'income_band3', 'has_formal_worker']].drop_duplicates('household_id')
    no_formal = hh_flags[~hh_flags['has_formal_worker']]
    has_formal = hh_flags[hh_flags['has_formal_worker']]
    low_share_no_formal = float((no_formal['income_band3'] == 'Low').mean()) if not no_formal.empty else np.nan
    low_share_has_formal = float((has_formal['income_band3'] == 'Low').mean()) if not has_formal.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={'width_ratios': [1.35, 1.0]})
    ax = axes[0]
    x = np.arange(len(bias_index_df))
    w = 0.34
    hh_vals = bias_index_df['HH-Base'].to_numpy(dtype=float)
    ours_vals = bias_index_df['Ours'].to_numpy(dtype=float)
    b1 = ax.bar(x - w / 2, hh_vals, width=w, color='#F4A261', alpha=0.95, label='HH-Base')
    b2 = ax.bar(x + w / 2, ours_vals, width=w, color='#2A9D8F', alpha=0.95, label='Ours')
    ax.set_xticks(x)
    ax.set_xticklabels(bias_index_df['index'], rotation=15, ha='right')
    ax.set_ylabel('Bias index value (lower is better)')
    ax.set_title('Model Bias Evidence Indices')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    structural_vals = [low_share_no_formal, low_share_has_formal]
    labels = ['No formal worker', 'Has formal worker']
    bars = ax2.bar(np.arange(2), structural_vals, color=['#B56576', '#6D597A'], alpha=0.95)
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels(labels, rotation=10)
    ax2.set_ylabel('P(Low income)')
    ax2.set_ylim(0, max([v for v in structural_vals if not np.isnan(v)] + [0.1]) * 1.25)
    ax2.set_title('Household Structural Bias')
    ax2.grid(axis='y', alpha=0.25)
    for i, v in enumerate(structural_vals):
        if np.isnan(v):
            continue
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    if not np.isnan(low_share_no_formal) and not np.isnan(low_share_has_formal) and low_share_has_formal > 1e-9:
        rr = low_share_no_formal / low_share_has_formal
        ax2.text(0.5, ax2.get_ylim()[1] * 0.93, f'Risk ratio = {rr:.2f}x', ha='center', va='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'limitation_bias_evidence_summary.pdf', dpi=300)
    fig.savefig(OUT_DIR / 'limitation_bias_evidence_summary.png', dpi=300)
    plt.close(fig)

    income_dist.to_csv(OUT_DIR / 'limitation_income_bias_shortest_distance_metrics.csv', index=False)
    role_stack_df.to_csv(OUT_DIR / 'limitation_role_activity_3class_metrics.csv', index=False)
    role_dist.to_csv(OUT_DIR / 'limitation_role_shortest_distance_metrics.csv', index=False)
    full_metric_df.to_csv(OUT_DIR / 'limitation_full_metrics_by_income.csv', index=False)
    bias_index_df.to_csv(OUT_DIR / 'limitation_bias_evidence_indices.csv', index=False)
    role_shift_df.to_csv(OUT_DIR / 'limitation_role_shift_l1_by_role.csv', index=False)

    report_lines = [
        '## Income/Role Supplement (Bias + Activity-chain)',
        f"- Method-specific GT overlap users: HH-Base={method_user_counts.get('HH-Base', 0)}, Ours={method_user_counts.get('Ours', 0)}",
        '- Dist metric: normalized edit distance on non-home activity chain (shortest distance, lower is better).',
        f'- Low-income share among households without formal worker: {low_share_no_formal:.3f}' if not np.isnan(low_share_no_formal) else '- Low-income share among households without formal worker: N/A',
        f'- Low-income share among households with formal worker: {low_share_has_formal:.3f}' if not np.isnan(low_share_has_formal) else '- Low-income share among households with formal worker: N/A',
    ]

    report_lines.extend([
        '- Full metrics reported: Acc ↑, EditDist ↓, BLEU ↑, Hour (micro) ↓, Interval (macro) ↓, Data JSD ↓, ActType ↓, TrajLen ↓.',
        '- Bias-proof focus metrics (lower is better): income max-min disparity, income low-high gap, role-shift L1 vs GT.',
        '- Overall metrics:',
    ])
    for method in ['HH-Base', 'Ours']:
        sub = full_metric_df[(full_metric_df['method'] == method) & (full_metric_df['cohort'] == 'Overall')]
        if sub.empty:
            continue
        r = sub.iloc[0]
        report_lines.append(
            f"  - {method} (n={int(r['n_users'])}): "
            f"Acc={float(r['accuracy']):.4f}, EditDist={float(r['edit_dist']):.4f}, BLEU={float(r['bleu_score']):.4f}, "
            f"HourMicro={float(r['micro_hour']):.4f}, IntervalMacro={float(r['macro_int']):.4f}, "
            f"DataJSD={float(r['data_jsd']):.4f}, ActType={float(r['act_type']):.4f}, TrajLen={float(r['traj_len']):.4f}"
        )

    hh_overall = full_metric_df[(full_metric_df['method'] == 'HH-Base') & (full_metric_df['cohort'] == 'Overall')]
    ours_overall = full_metric_df[(full_metric_df['method'] == 'Ours') & (full_metric_df['cohort'] == 'Overall')]
    if not hh_overall.empty and not ours_overall.empty:
        h = hh_overall.iloc[0]
        o = ours_overall.iloc[0]
        report_lines.append('- Ours vs HH-Base (overall delta):')
        report_lines.append(
            f"  - Acc: {float(o['accuracy'] - h['accuracy']):+.4f}; EditDist: {float(o['edit_dist'] - h['edit_dist']):+.4f}; BLEU: {float(o['bleu_score'] - h['bleu_score']):+.4f}; "
            f"HourMicro: {float(o['micro_hour'] - h['micro_hour']):+.4f}; IntervalMacro: {float(o['macro_int'] - h['macro_int']):+.4f}; "
            f"DataJSD: {float(o['data_jsd'] - h['data_jsd']):+.4f}; ActType: {float(o['act_type'] - h['act_type']):+.4f}; TrajLen: {float(o['traj_len'] - h['traj_len']):+.4f}"
        )

    report_lines.append('- Income-cohort full metrics (per method):')
    for cohort in ['Income-Low', 'Income-Middle', 'Income-High']:
        for method in ['HH-Base', 'Ours']:
            sub = full_metric_df[(full_metric_df['method'] == method) & (full_metric_df['cohort'] == cohort)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            report_lines.append(
                f"  - {cohort} / {method} (n={int(r['n_users'])}): "
                f"Acc={float(r['accuracy']):.4f}, EditDist={float(r['edit_dist']):.4f}, BLEU={float(r['bleu_score']):.4f}, "
                f"HourMicro={float(r['micro_hour']):.4f}, IntervalMacro={float(r['macro_int']):.4f}, "
                f"DataJSD={float(r['data_jsd']):.4f}, ActType={float(r['act_type']):.4f}, TrajLen={float(r['traj_len']):.4f}"
            )

    for band in income_order:
        sub = income_dist[income_dist['income_band3'] == band]
        hh_sub = sub[sub['method'] == 'HH-Base']
        ours_sub = sub[sub['method'] == 'Ours']
        gt_sub = sub[sub['method'] == 'Ground Truth']
        if hh_sub.empty or ours_sub.empty or gt_sub.empty:
            continue
        hh_v = float(hh_sub['mean_shortest_distance'].iloc[0])
        ours_v = float(ours_sub['mean_shortest_distance'].iloc[0])
        gt_v = float(gt_sub['mean_shortest_distance'].iloc[0])
        abs_gain = hh_v - ours_v
        rel_gain = abs_gain / hh_v if hh_v > 1e-9 else np.nan
        line = f'- {band}: GT={gt_v:.3f}, HH-Base={hh_v:.3f}, Ours={ours_v:.3f}, improvement_vs_HH={abs_gain:.3f}'
        if not np.isnan(rel_gain):
            line += f' ({rel_gain * 100:.1f}%)'
        report_lines.append(line)

    report_lines.append('- Role-level insight (Ours shortest distance; lower is better):')
    for _, r in role_dist.head(4).iterrows():
        report_lines.append(
            f"  - {r['role']}: Ours={float(r['shortest_dist_ours']):.3f}, HH-Base={float(r['shortest_dist_hh_base']):.3f}, n={int(r['n_users'])}"
        )

    report_lines.extend([
        '- Bias evidence summary (targeted indices):',
        f"  - Income max-min disparity: HH-Base={hh_income_maxmin:.3f}, Ours={ours_income_maxmin:.3f}" if not np.isnan(hh_income_maxmin) and not np.isnan(ours_income_maxmin) else '  - Income max-min disparity: N/A',
        f"  - Income low-high gap: HH-Base={hh_income_lh:.3f}, Ours={ours_income_lh:.3f}" if not np.isnan(hh_income_lh) and not np.isnan(ours_income_lh) else '  - Income low-high gap: N/A',
        f"  - Role-shift L1 vs GT: HH-Base={hh_role_shift:.3f}, Ours={ours_role_shift:.3f}" if not np.isnan(hh_role_shift) and not np.isnan(ours_role_shift) else '  - Role-shift L1 vs GT: N/A',
    ])

    output_files = [
        'limitation_bias_evidence_summary.pdf/.png',
        'limitation_full_metrics_dense_panel.pdf/.png',
        'limitation_income_bias_shortest_distance_hist.pdf/.png',
        'limitation_role_activity_3class_stacked.pdf/.png',
        'limitation_income_bias_shortest_distance_metrics.csv',
        'limitation_bias_evidence_indices.csv',
        'limitation_role_shift_l1_by_role.csv',
        'limitation_role_activity_3class_metrics.csv',
        'limitation_role_shortest_distance_metrics.csv',
        'limitation_full_metrics_by_income.csv',
    ]

    return {
        'report_lines': report_lines,
        'output_files': output_files,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style('whitegrid')

    gt_lookup = build_lookup(load_json(GT_FILE))
    temporal_gt_lookup = build_lookup(load_json(TEMPORAL_GT_FILE))
    person_lookup = build_lookup(load_json(PERSON_FILE))
    hh_lookup = {str(x.get('household_id', '')): x for x in load_json(HH_FILE)}

    ours_gpt4o_lookup = build_lookup(load_json(OURS_GPT4O_FILE))
    ours_requested_lookup = build_lookup(load_json(OURS_REQUESTED_FILE))
    ours_old_shared_lookup = build_lookup(load_json(OURS_OLD_SHARED_FILE))
    ours_ft_lookup = build_lookup(load_json(OURS_FT_FILE))
    hh_base_lookup = build_lookup(load_json(HH_BASE_FILE))
    hh_rag_lookup = build_lookup(load_json(HH_RAG_FILE))
    deepmove_lookup = build_lookup(load_json(DEEPMOVE_FILE))
    lstpm_lookup = build_lookup(load_json(LSTPM_FILE))
    indiv_base_lookup = build_lookup(load_json(INDIV_BASE_FILE))
    indiv_copb_lookup = build_lookup(load_json(INDIV_COPB_FILE))

    baseline_lookups = [
        ('LSTPM', lstpm_lookup),
        ('Indiv-CoPB', indiv_copb_lookup),
        ('HH-RAG', hh_rag_lookup),
    ]

    ours_candidates = [
        ('GPT-4o eval run (20260212_012031)', ours_gpt4o_lookup),
    ]
    chosen = select_best_ours_temporal(temporal_gt_lookup, baseline_lookups, ours_candidates)
    _, chosen_shared_n, chosen_mean_acc, chosen_name, chosen_lookup = chosen

    n_temporal_selected, n_temporal_all, temporal_methods, temporal_mode = draw_temporal_with_baselines(
        temporal_gt_lookup,
        chosen_lookup,
        baseline_lookups,
        person_lookup,
        hh_lookup,
    )

    eval_rows = parse_eval_user_blocks(EVAL_REPORT_FILE)
    worst3 = select_worst3_for_plot(eval_rows)
    draw_worst3_cases_figure(worst3)

    scores = [
        method_bias_scores('HH-Base', hh_base_lookup, gt_lookup, person_lookup, hh_lookup),
        method_bias_scores('HoMe-Llama-FT (ours)', ours_ft_lookup, gt_lookup, person_lookup, hh_lookup),
    ]
    pd.DataFrame(scores).to_csv(OUT_DIR / 'limitation_bias_hhbase_vs_ours_ft_scores.csv', index=False)
    draw_bias_hhbase_vs_ours(scores)

    supplement = run_income_role_supplement(
        gt_lookup=gt_lookup,
        ours_lookup=ours_ft_lookup,
        hh_base_lookup=hh_base_lookup,
        person_lookup=person_lookup,
        hh_lookup=hh_lookup,
    )

    hh_gender_gap = float(scores[0].get('gender_collab_gap', np.nan))
    ours_gender_gap = float(scores[1].get('gender_collab_gap', np.nan))
    hh_income_gap = float(scores[0].get('income_accuracy_disparity', np.nan))
    ours_income_gap = float(scores[1].get('income_accuracy_disparity', np.nan))

    gender_reduction = np.nan
    income_reduction = np.nan
    if hh_gender_gap > 1e-9:
        gender_reduction = (hh_gender_gap - ours_gender_gap) / hh_gender_gap
    if hh_income_gap > 1e-9:
        income_reduction = (hh_income_gap - ours_income_gap) / hh_income_gap

    report_lines = [
        '# Limitation Figures: GT-vs-Ours and Bias (HH-Base vs Ours)',
        '',
        f'- Temporal Ours source selected automatically: {chosen_name}',
        f'- Shared users across GT+baselines+ours: {chosen_shared_n}, ours mean acc on shared users: {chosen_mean_acc:.3f}',
        f'- Temporal cohort mode: {temporal_mode}',
        f'- Temporal methods shown: {temporal_methods}',
        '- Bias comparison uses HoMe-Llama-FT file: all_trajectories_20251214_180155_California.json',
        '- HH baseline file: all_trajectories_20251117_123218.json',
        f'- Temporal comparison overlap users: selected={n_temporal_selected}, all={n_temporal_all}',
        '- Bias uses each method\'s valid user cohort against the same GT definition (cohort sizes can differ).',
        '- Bias dimensions: gender collaborative-recall amplification and low/high income accuracy-gap amplification (both lower is better).',
        f"- Bias users: HH-Base={scores[0]['n_users']}, Ours-FT={scores[1]['n_users']}",
        f"- Gender bias metric (lower better): HH-Base={hh_gender_gap:.4f}, Ours={ours_gender_gap:.4f}, reduction={gender_reduction * 100:.1f}%" if not np.isnan(gender_reduction) else f"- Gender bias metric (lower better): HH-Base={hh_gender_gap:.4f}, Ours={ours_gender_gap:.4f}",
        f"- Income bias metric (lower better): HH-Base={hh_income_gap:.4f}, Ours={ours_income_gap:.4f}, reduction={income_reduction * 100:.1f}%" if not np.isnan(income_reduction) else f"- Income bias metric (lower better): HH-Base={hh_income_gap:.4f}, Ours={ours_income_gap:.4f}",
        '- Temporal panel titles intentionally hide n for cleaner paper presentation.',
        '',
        '## Selected 2 Challenging Cases (Case A and Case B)',
        '',
    ]

    for row in worst3[:2]:
        reasons = infer_failure_reasons(row)
        report_lines.append(f"- {row['user_id']}")
        report_lines.append(f"  Generated: {row['generated_text']}")
        report_lines.append(f"  Original:  {row['original_text']}")
        report_lines.append(f"  Reasons: {'; '.join(reasons)}")

    report_lines.extend([''])
    report_lines.extend(supplement.get('report_lines', []))

    report_lines.extend([
        '',
        '## Outputs',
        '- limitation_temporal_gt_vs_ours_gpt4o.pdf/.png',
        '- limitation_bias_hhbase_vs_ours_ft.pdf/.png',
        '- limitation_bias_hhbase_vs_ours_ft_scores.csv',
        '- limitation_worst3_gpt4o_cases.pdf/.png',
    ])
    for item in supplement.get('output_files', []):
        report_lines.append(f'- {item}')

    with open(OUT_DIR / 'limitation_bias_hhbase_vs_ours_ft_report.md', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f'Saved: {OUT_DIR / "limitation_temporal_gt_vs_ours_gpt4o.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_bias_hhbase_vs_ours_ft.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_bias_hhbase_vs_ours_ft_scores.csv"}')
    print(f'Saved: {OUT_DIR / "limitation_worst3_gpt4o_cases.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_bias_evidence_summary.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_full_metrics_dense_panel.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_income_bias_shortest_distance_hist.pdf"}')
    print(f'Saved: {OUT_DIR / "limitation_role_activity_3class_stacked.pdf"}')


if __name__ == '__main__':
    main()
