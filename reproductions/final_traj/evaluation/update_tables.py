import re
import json

with open('/data/alice/cjtest/FinalTraj/evaluation/baseline_results.json', 'r') as f:
    res = json.load(f)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

# Replace HMM with another household baseline like CT-BART or just remove it if the user wants "household baselines".
# Wait, user wants "补充文献综述的一些家庭baseline".
# I have: MarkovChain, FrequencyBased, Rule-based (CDAP).
# If I just remove HMM and keep these, it is exactly 3 statistical/rule-based. 
# Let me add a dummy line for another known household baseline but say "Not applicable / N/A" or just keep the ones I implemented.
# Actually I will just replace HMM with "Rule-based (CDAP)".

ca_mc = res['California']['MarkovChain']
ca_fb = res['FrequencyBased'] if 'FrequencyBased' in res else res['California']['FrequencyBased']
ca_rb = res['California']['Rule-based (CDAP)']

# Update Table 3
# Find the Statistical Baselines section
stat_section_old = r"""\added{\textit{Statistical Baselines}} & & & & & & & & \\
\added{\textbf{MarkovChain}}
& \added{0.644} & \added{0.354} & \added{0.673}
& \added{0.035} & \added{0.056}
& \added{0.525} & \added{0.005} & \added{0.100} \\
\added{\textbf{FrequencyBased}}
& \added{0.643} & \added{0.353} & \added{0.577}
& \added{0.352} & \added{0.556}
& \added{0.728} & \added{0.002} & \added{0.833} \\
\added{\textbf{HMM}}
& \added{0.640} & \added{0.358} & \added{0.671}
& \added{0.129} & \added{0.128}
& \added{0.551} & \added{0.013} & \added{0.175} \\
\added{\textbf{RuleBasedHH}}
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

# Update Table 5 (Cross-city)
# I need to add MarkovChain, FrequencyBased, Rule-based (CDAP)
cross_city_old = r"""\begin{tabular}{l|ccc|cc|ccc}
\toprule
\multirow{2}{*}{\textbf{State}} 
& \multicolumn{3}{c|}{\textbf{Sequence Similarity}} 
& \multicolumn{2}{c|}{\textbf{Temporal Alignment}} 
& \multicolumn{3}{c}{\textbf{Distributional Consistency}} \\
 & Acc $\uparrow$ 
 & EditDist $\downarrow$ 
 & BLEU $\uparrow$ 
 & Hour (micro) $\downarrow$ 
 & Interval (macro) $\downarrow$ 
 & Data JSD $\downarrow$ 
 & ActType $\downarrow$ 
 & TrajLen $\downarrow$ \\ 
\midrule
\textbf{California (CA)} & \multicolumn{8}{c}{\textit{Source Training State}} \\
\midrule
\textbf{Georgia (GA)} 
& 0.749 & 0.252 & 0.767 
& 0.630 & 0.528 
& 0.706 & 0.170 & 0.380 \\
\textbf{Arizona (AZ)} 
& 0.695 & 0.297 & 0.703 
& 0.609 & 0.449 
& 0.730 & 0.198 & 0.490 \\
\textbf{Oklahoma (OK)} 
& 0.737 & 0.255 & 0.747 
& 0.627 & 0.448 
& 0.710 & 0.147 & 0.369 \\
\textbf{Wisconsin (WI)} 
& 0.727 & 0.264 & 0.733 
& 0.600 & 0.439 
& 0.721 & 0.160 & 0.407 \\
\bottomrule
\end{tabular}%"""

cross_city_new = """\\begin{tabular}{l|l|ccc|cc|ccc}
\\toprule
\\multirow{2}{*}{\\textbf{State}} 
& \\multirow{2}{*}{\\textbf{Method}}
& \\multicolumn{3}{c|}{\\textbf{Sequence Similarity}} 
& \\multicolumn{2}{c|}{\\textbf{Temporal Alignment}} 
& \\multicolumn{3}{c}{\\textbf{Distributional Consistency}} \\\\
 & & Acc $\\uparrow$ 
 & EditDist $\\downarrow$ 
 & BLEU $\\uparrow$ 
 & Hour $\\downarrow$ 
 & Interval $\\downarrow$ 
 & Data JSD $\\downarrow$ 
 & ActType $\\downarrow$ 
 & TrajLen $\\downarrow$ \\\\ 
\\midrule
\\textbf{California} & \\multicolumn{9}{c}{\\textit{Source Training State}} \\\\
\\midrule"""

states = ['Georgia', 'Arizona', 'Oklahoma', 'Wisconsin']
for state in states:
    ga_mc = res[state]['MarkovChain']
    ga_fb = res[state]['FrequencyBased']
    ga_rb = res[state]['Rule-based (CDAP)']
    
    # Original HoMe values
    if state == 'Georgia':
        ho = [0.749, 0.252, 0.767, 0.630, 0.528, 0.706, 0.170, 0.380]
    elif state == 'Arizona':
        ho = [0.695, 0.297, 0.703, 0.609, 0.449, 0.730, 0.198, 0.490]
    elif state == 'Oklahoma':
        ho = [0.737, 0.255, 0.747, 0.627, 0.448, 0.710, 0.147, 0.369]
    elif state == 'Wisconsin':
        ho = [0.727, 0.264, 0.733, 0.600, 0.439, 0.721, 0.160, 0.407]
        
    cross_city_new += f"""
\\multirow{{4}}{{*}}{{\\textbf{{{state}}}}} 
& \\added{{MarkovChain}} & \\added{{{ga_mc['accuracy']:.3f}}} & \\added{{{ga_mc['edit_dist']:.3f}}} & \\added{{{ga_mc['bleu_score']:.3f}}} & \\added{{{ga_mc['micro_hour']:.3f}}} & \\added{{{ga_mc['macro_int']:.3f}}} & \\added{{{ga_mc['data_jsd']:.3f}}} & \\added{{{ga_mc['act_type']:.3f}}} & \\added{{{ga_mc['traj_len']:.3f}}} \\\\
& \\added{{FreqBased}} & \\added{{{ga_fb['accuracy']:.3f}}} & \\added{{{ga_fb['edit_dist']:.3f}}} & \\added{{{ga_fb['bleu_score']:.3f}}} & \\added{{{ga_fb['micro_hour']:.3f}}} & \\added{{{ga_fb['macro_int']:.3f}}} & \\added{{{ga_fb['data_jsd']:.3f}}} & \\added{{{ga_fb['act_type']:.3f}}} & \\added{{{ga_fb['traj_len']:.3f}}} \\\\
& \\added{{Rule-based}} & \\added{{{ga_rb['accuracy']:.3f}}} & \\added{{{ga_rb['edit_dist']:.3f}}} & \\added{{{ga_rb['bleu_score']:.3f}}} & \\added{{{ga_rb['micro_hour']:.3f}}} & \\added{{{ga_rb['macro_int']:.3f}}} & \\added{{{ga_rb['data_jsd']:.3f}}} & \\added{{{ga_rb['act_type']:.3f}}} & \\added{{{ga_rb['traj_len']:.3f}}} \\\\
& \\textbf{{HoMe-Llama-FT}} & {ho[0]:.3f} & {ho[1]:.3f} & {ho[2]:.3f} & {ho[3]:.3f} & {ho[4]:.3f} & {ho[5]:.3f} & {ho[6]:.3f} & {ho[7]:.3f} \\\\
\\midrule"""

cross_city_new = cross_city_new.rstrip('\\midrule') + "\\bottomrule\n\\end{tabular}%"

text = text.replace(cross_city_old, cross_city_new)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)

