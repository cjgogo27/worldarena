import json

with open('/data/alice/cjtest/FinalTraj/evaluation/baseline_results.json', 'r') as f:
    res = json.load(f)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

ca_mc = res['California']['MarkovChain']
ca_fb = res['FrequencyBased'] if 'FrequencyBased' in res else res['California']['FrequencyBased']
ca_rb = res['California']['Rule-based (CDAP)']

stat_section_old = r"""\added{\textit{Statistical \& Rule-Based}} & & & & & & & & \\
\added{\textbf{MarkovChain}}
& \added{0.644} & \added{0.354} & \added{0.673}
& \added{0.035} & \added{0.056}
& \added{0.525} & \added{0.005} & \added{0.100} \\
\added{\textbf{FrequencyBased}}
& \added{0.643} & \added{0.353} & \added{0.577}
& \added{0.352} & \added{0.556}
& \added{0.728} & \added{0.002} & \added{0.833} \\
\added{\textbf{Rule-based (CDAP)}}
& \added{0.753} & \added{0.243} & \added{0.793}
& \added{0.604} & \added{0.422}
& \added{0.698} & \added{0.100} & \added{0.455} \\"""

stat_section_new = f"""\\added{{\\textit{{Statistical \& Rule-Based}}}} & & & & & & & & \\\\
\\added{{\\textbf{{MarkovChain}}}}
& \\added{{{ca_mc['accuracy']:.3f}}} & \\added{{{ca_mc['edit_dist']:.3f}}} & \\added{{{ca_mc['bleu_score']:.3f}}}
& \\added{{{ca_mc['micro_hour']:.3f}}} & \\added{{{ca_mc['macro_int']:.3f}}}
& \\added{{{ca_mc['data_jsd']:.3f}}} & \\added{{{ca_mc['act_type']:.3f}}} & \\added{{{ca_mc['traj_len']:.3f}}} \\\\
\\added{{\\textbf{{FrequencyBased}}}}
& \\added{{{ca_fb['accuracy']:.3f}}} & \\added{{{ca_fb['edit_dist']:.3f}}} & \\added{{{ca_fb['bleu_score']:.3f}}}
& \\added{{{ca_fb['micro_hour']:.3f}}} & \\added{{{ca_fb['macro_int']:.3f}}}
& \\added{{{ca_fb['data_jsd']:.3f}}} & \\added{{{ca_fb['act_type']:.3f}}} & \\added{{{ca_fb['traj_len']:.3f}}} \\\\
\\added{{\\textbf{{Rule-based (CDAP)}}}}
& \\added{{{ca_rb['accuracy']:.3f}}} & \\added{{{ca_rb['edit_dist']:.3f}}} & \\added{{{ca_rb['bleu_score']:.3f}}}
& \\added{{{ca_rb['micro_hour']:.3f}}} & \\added{{{ca_rb['macro_int']:.3f}}}
& \\added{{{ca_rb['data_jsd']:.3f}}} & \\added{{{ca_rb['act_type']:.3f}}} & \\added{{{ca_rb['traj_len']:.3f}}} \\\\"""

text = text.replace(stat_section_old, stat_section_new)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)

