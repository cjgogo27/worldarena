import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


OUT_DIR = Path('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/fig')

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
ACT_EDGE_WIDTH = 0.55

METHOD_ORDER = [
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

DISPLAY_LABELS = {
    'Ground Truth': 'Ground Truth',
    'DeepMove': 'DeepMove',
    'LSTPM': 'LSTPM',
    'MarkovChain': 'MarkovChain',
    'Empirical Sampling': 'Empirical',
    'Rule-based (CDAP)': 'CDAP',
    'Indiv-Base': 'Indiv-Base',
    'Indiv-CoPB': 'Indiv-CoPB',
    'HH-Base': 'HH-Base',
    'HH-RAG': 'HH-RAG',
    'HoMe-LLM': 'HoMe-LLM',
}

DATA = {
    'Ground Truth': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '13:40'},
                {'activity': 'socialize', 'start_time': '13:40', 'end_time': '15:40'},
                {'activity': 'home', 'start_time': '15:40', 'end_time': '17:40'},
                {'activity': 'socialize', 'start_time': '17:40', 'end_time': '19:40'},
                {'activity': 'home', 'start_time': '19:40', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:00'},
                {'activity': 'work', 'start_time': '09:00', 'end_time': '12:25'},
                {'activity': 'dine_out', 'start_time': '12:25', 'end_time': '12:40'},
                {'activity': 'work', 'start_time': '12:40', 'end_time': '20:40'},
                {'activity': 'home', 'start_time': '20:40', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '11:50'},
                {'activity': 'work', 'start_time': '11:50', 'end_time': '15:40'},
                {'activity': 'home', 'start_time': '15:40', 'end_time': '24:00'},
            ],
        },
    ],
    'DeepMove': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:06'},
                {'activity': 'work', 'start_time': '08:06', 'end_time': '11:02'},
                {'activity': 'home', 'start_time': '11:02', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:11'},
                {'activity': 'service', 'start_time': '08:11', 'end_time': '08:54'},
                {'activity': 'work', 'start_time': '08:54', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '07:48'},
                {'activity': 'service', 'start_time': '07:48', 'end_time': '08:29'},
                {'activity': 'medical', 'start_time': '08:29', 'end_time': '11:56'},
                {'activity': 'work', 'start_time': '11:56', 'end_time': '20:01'},
                {'activity': 'home', 'start_time': '20:01', 'end_time': '22:06'},
                {'activity': 'work', 'start_time': '22:06', 'end_time': '24:00'},
            ],
        },
    ],
    'LSTPM': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '10:20'},
                {'activity': 'service', 'start_time': '10:20', 'end_time': '11:30'},
                {'activity': 'home', 'start_time': '11:30', 'end_time': '14:10'},
                {'activity': 'work', 'start_time': '14:10', 'end_time': '17:40'},
                {'activity': 'home', 'start_time': '17:40', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:40'},
                {'activity': 'work', 'start_time': '06:40', 'end_time': '11:05'},
                {'activity': 'home', 'start_time': '11:05', 'end_time': '15:30'},
                {'activity': 'shopping', 'start_time': '15:30', 'end_time': '16:15'},
                {'activity': 'work', 'start_time': '16:15', 'end_time': '19:20'},
                {'activity': 'home', 'start_time': '19:20', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:30'},
                {'activity': 'service', 'start_time': '09:30', 'end_time': '10:25'},
                {'activity': 'work', 'start_time': '10:25', 'end_time': '13:50'},
                {'activity': 'home', 'start_time': '13:50', 'end_time': '17:10'},
                {'activity': 'work', 'start_time': '17:10', 'end_time': '19:05'},
                {'activity': 'home', 'start_time': '19:05', 'end_time': '24:00'},
            ],
        },
    ],
    'MarkovChain': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '07:52'},
                {'activity': 'work', 'start_time': '07:52', 'end_time': '17:49'},
                {'activity': 'home', 'start_time': '17:49', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:58'},
                {'activity': 'work', 'start_time': '08:58', 'end_time': '17:51'},
                {'activity': 'home', 'start_time': '17:51', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:16'},
                {'activity': 'work', 'start_time': '06:16', 'end_time': '19:29'},
                {'activity': 'home', 'start_time': '19:29', 'end_time': '24:00'},
            ],
        },
    ],
    'Empirical Sampling': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '07:19'},
                {'activity': 'socialize', 'start_time': '07:19', 'end_time': '08:39'},
                {'activity': 'home', 'start_time': '08:39', 'end_time': '10:04'},
                {'activity': 'medical', 'start_time': '10:04', 'end_time': '10:39'},
                {'activity': 'home', 'start_time': '10:39', 'end_time': '11:10'},
                {'activity': 'service', 'start_time': '11:10', 'end_time': '11:27'},
                {'activity': 'home', 'start_time': '11:27', 'end_time': '11:52'},
                {'activity': 'socialize', 'start_time': '11:52', 'end_time': '13:42'},
                {'activity': 'medical', 'start_time': '13:42', 'end_time': '14:44'},
                {'activity': 'socialize', 'start_time': '14:44', 'end_time': '16:25'},
                {'activity': 'home', 'start_time': '16:25', 'end_time': '17:17'},
                {'activity': 'service', 'start_time': '17:17', 'end_time': '17:49'},
                {'activity': 'home', 'start_time': '17:49', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '10:02'},
                {'activity': 'work', 'start_time': '10:02', 'end_time': '16:51'},
                {'activity': 'home', 'start_time': '16:51', 'end_time': '21:12'},
                {'activity': 'service', 'start_time': '21:12', 'end_time': '21:43'},
                {'activity': 'home', 'start_time': '21:43', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '07:58'},
                {'activity': 'work', 'start_time': '07:58', 'end_time': '14:10'},
                {'activity': 'home', 'start_time': '14:10', 'end_time': '14:58'},
                {'activity': 'shopping', 'start_time': '14:58', 'end_time': '15:52'},
                {'activity': 'home', 'start_time': '15:52', 'end_time': '17:01'},
                {'activity': 'medical', 'start_time': '17:01', 'end_time': '17:29'},
                {'activity': 'home', 'start_time': '17:29', 'end_time': '24:00'},
            ],
        },
    ],
    'Rule-based (CDAP)': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:01'},
                {'activity': 'socialize', 'start_time': '09:01', 'end_time': '13:02'},
                {'activity': 'home', 'start_time': '13:02', 'end_time': '18:23'},
                {'activity': 'medical', 'start_time': '18:23', 'end_time': '18:48'},
                {'activity': 'home', 'start_time': '18:48', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:06'},
                {'activity': 'work', 'start_time': '09:06', 'end_time': '12:59'},
                {'activity': 'home', 'start_time': '12:59', 'end_time': '13:21'},
                {'activity': 'work', 'start_time': '13:21', 'end_time': '20:08'},
                {'activity': 'home', 'start_time': '20:08', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:07'},
                {'activity': 'service', 'start_time': '08:07', 'end_time': '09:59'},
                {'activity': 'work', 'start_time': '09:59', 'end_time': '16:26'},
                {'activity': 'home', 'start_time': '16:26', 'end_time': '20:34'},
                {'activity': 'service', 'start_time': '20:34', 'end_time': '20:56'},
                {'activity': 'home', 'start_time': '20:56', 'end_time': '24:00'},
            ],
        },
    ],
    'Indiv-Base': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:48'},
                {'activity': 'exercise', 'start_time': '06:48', 'end_time': '07:29'},
                {'activity': 'home', 'start_time': '07:29', 'end_time': '12:14'},
                {'activity': 'socialize', 'start_time': '12:14', 'end_time': '14:58'},
                {'activity': 'dine_out', 'start_time': '14:58', 'end_time': '15:22'},
                {'activity': 'home', 'start_time': '15:22', 'end_time': '17:07'},
                {'activity': 'service', 'start_time': '17:07', 'end_time': '17:41'},
                {'activity': 'socialize', 'start_time': '17:41', 'end_time': '19:23'},
                {'activity': 'home', 'start_time': '19:23', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:05'},
                {'activity': 'work', 'start_time': '08:05', 'end_time': '12:18'},
                {'activity': 'dine_out', 'start_time': '12:18', 'end_time': '13:03'},
                {'activity': 'work', 'start_time': '13:03', 'end_time': '19:34'},
                {'activity': 'shopping', 'start_time': '19:34', 'end_time': '19:59'},
                {'activity': 'home', 'start_time': '19:59', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:46'},
                {'activity': 'work', 'start_time': '09:46', 'end_time': '15:07'},
                {'activity': 'service', 'start_time': '15:07', 'end_time': '15:36'},
                {'activity': 'home', 'start_time': '15:36', 'end_time': '24:00'},
            ],
        },
    ],
    'Indiv-CoPB': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '11:20'},
                {'activity': 'socialize', 'start_time': '11:20', 'end_time': '13:05'},
                {'activity': 'home', 'start_time': '13:05', 'end_time': '18:10'},
                {'activity': 'service', 'start_time': '18:10', 'end_time': '18:40'},
                {'activity': 'home', 'start_time': '18:40', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:20'},
                {'activity': 'work', 'start_time': '09:20', 'end_time': '12:00'},
                {'activity': 'home', 'start_time': '12:00', 'end_time': '14:50'},
                {'activity': 'work', 'start_time': '14:50', 'end_time': '19:10'},
                {'activity': 'home', 'start_time': '19:10', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '10:45'},
                {'activity': 'work', 'start_time': '10:45', 'end_time': '14:45'},
                {'activity': 'home', 'start_time': '14:45', 'end_time': '16:30'},
                {'activity': 'service', 'start_time': '16:30', 'end_time': '17:10'},
                {'activity': 'home', 'start_time': '17:10', 'end_time': '24:00'},
            ],
        },
    ],
    'HH-Base': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:56'},
                {'activity': 'home', 'start_time': '06:56', 'end_time': '07:41'},
                {'activity': 'socialize', 'start_time': '07:41', 'end_time': '14:19'},
                {'activity': 'dine_out', 'start_time': '14:19', 'end_time': '14:47'},
                {'activity': 'medical', 'start_time': '14:47', 'end_time': '15:26'},
                {'activity': 'home', 'start_time': '15:26', 'end_time': '16:08'},
                {'activity': 'service', 'start_time': '16:08', 'end_time': '16:33'},
                {'activity': 'home', 'start_time': '16:33', 'end_time': '17:02'},
                {'activity': 'medical', 'start_time': '17:02', 'end_time': '17:23'},
                {'activity': 'home', 'start_time': '17:23', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:47'},
                {'activity': 'home', 'start_time': '06:47', 'end_time': '07:29'},
                {'activity': 'work', 'start_time': '07:29', 'end_time': '17:57'},
                {'activity': 'service', 'start_time': '17:57', 'end_time': '18:23'},
                {'activity': 'home', 'start_time': '18:23', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '06:39'},
                {'activity': 'home', 'start_time': '06:39', 'end_time': '07:22'},
                {'activity': 'work', 'start_time': '07:22', 'end_time': '17:06'},
                {'activity': 'dine_out', 'start_time': '17:06', 'end_time': '17:28'},
                {'activity': 'service', 'start_time': '17:28', 'end_time': '18:15'},
                {'activity': 'home', 'start_time': '18:15', 'end_time': '24:00'},
            ],
        },
    ],
    'HH-RAG': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:10'},
                {'activity': 'socialize', 'start_time': '09:10', 'end_time': '12:40'},
                {'activity': 'home', 'start_time': '12:40', 'end_time': '16:10'},
                {'activity': 'socialize', 'start_time': '16:10', 'end_time': '17:35'},
                {'activity': 'home', 'start_time': '17:35', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:50'},
                {'activity': 'work', 'start_time': '08:50', 'end_time': '12:10'},
                {'activity': 'home', 'start_time': '12:10', 'end_time': '15:10'},
                {'activity': 'work', 'start_time': '15:10', 'end_time': '18:20'},
                {'activity': 'home', 'start_time': '18:20', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '09:05'},
                {'activity': 'work', 'start_time': '09:05', 'end_time': '14:30'},
                {'activity': 'home', 'start_time': '14:30', 'end_time': '16:40'},
                {'activity': 'work', 'start_time': '16:40', 'end_time': '18:05'},
                {'activity': 'home', 'start_time': '18:05', 'end_time': '24:00'},
            ],
        },
    ],
    'HoMe-LLM': [
        {
            'user_id': '30339694_1',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '13:28'},
                {'activity': 'socialize', 'start_time': '13:28', 'end_time': '14:36'},
                {'activity': 'home', 'start_time': '14:36', 'end_time': '17:33'},
                {'activity': 'socialize', 'start_time': '17:33', 'end_time': '19:28'},
                {'activity': 'home', 'start_time': '19:28', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_2',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '08:41'},
                {'activity': 'work', 'start_time': '08:41', 'end_time': '12:18'},
                {'activity': 'dine_out', 'start_time': '12:18', 'end_time': '12:53'},
                {'activity': 'work', 'start_time': '12:53', 'end_time': '20:27'},
                {'activity': 'home', 'start_time': '20:27', 'end_time': '24:00'},
            ],
        },
        {
            'user_id': '30339694_3',
            'schedule': [
                {'activity': 'home', 'start_time': '00:00', 'end_time': '10:07'},
                {'activity': 'work', 'start_time': '10:07', 'end_time': '16:04'},
                {'activity': 'service', 'start_time': '16:04', 'end_time': '16:29'},
                {'activity': 'home', 'start_time': '16:29', 'end_time': '24:00'},
            ],
        },
    ],
}


def to_min(s):
    hh, mm = s.split(':')
    return int(hh) * 60 + int(mm)


def draw_row(ax, schedule):
    for seg in schedule:
        st = to_min(seg['start_time'])
        ed = to_min(seg['end_time'])
        act = seg['activity']
        color = ACT_COLORS.get(act, '#CCCCCC')
        ax.barh(0.5, ed - st, left=st, height=0.76, color=color, edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    users = [x['user_id'] for x in DATA['Ground Truth']]
    n_rows = len(METHOD_ORDER)
    n_cols = len(users)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 1.40 * n_rows), sharex=True)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for c, uid in enumerate(users):
        for r, method in enumerate(METHOD_ORDER):
            ax = axes[r, c]
            row = next(x for x in DATA[method] if x['user_id'] == uid)
            draw_row(ax, row['schedule'])

            if r == 0:
                ax.set_title(f'Member{c + 1}', fontsize=17, pad=11, fontweight='bold')

            ax.set_xlim(0, 1440)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 360, 720, 1080, 1440])
            if r != n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'], fontsize=14)
            ax.grid(axis='x', alpha=0.16)

            if c == 0:
                ax.set_ylabel(DISPLAY_LABELS.get(method, method), rotation=0, labelpad=74, va='center', fontsize=16)

    handles = [mpatches.Patch(facecolor=ACT_COLORS[k], edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH, label=k) for k in [
        'home', 'work', 'socialize', 'dine_out', 'service', 'shopping', 'medical', 'exercise', 'education', 'dropoff_pickup'
    ]]
    fig.legend(handles=handles, loc='center right', frameon=False, fontsize=13)

    fig.text(0.5, 0.012, 'Time of Day', ha='center', fontsize=16)
    fig.tight_layout(rect=[0.06, 0.03, 0.89, 0.99])

    out_pdf = OUT_DIR / 'limitation_method_timeline_comparison.pdf'
    out_png = OUT_DIR / 'limitation_method_timeline_comparison.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)


if __name__ == '__main__':
    main()
