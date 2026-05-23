import json

with open('/data/alice/cjtest/FinalTraj/evaluation/baseline_results.json', 'r') as f:
    res = json.load(f)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

old_table = r"""\begin{tabular}{l|ccc|cc|ccc}
\toprule
\multirow{2}{*}{\textbf{State}} 
& \multicolumn{3}{c|}{\textbf{Sequence Similarity}} 
& \multicolumn{2}{c|}{\textbf{Temporal Alignment}} 
& \multicolumn{3}{c}{\textbf{Distributional Consistency}} \\
 & Acc
 & EditDist
 & BLEU
 & Hour (micro)
 & Interval (macro)
 & Data JSD
 & ActType
 & TrajLen \\ 
\midrule
California & 0.752 & 0.245 & 0.778 & 0.600 & 0.307 & 0.718 & 0.183 & 0.272 \\
Georgia   & 0.749 & 0.249 & 0.809 & 0.557 & 0.343 & 0.734 & 0.156 & 0.298 \\
Arizona   & 0.695 & 0.302 & 0.745 & 0.559 & 0.333 & 0.738 & 0.192 & 0.256 \\
Oklahoma  & 0.737 & 0.262 & 0.761 & 0.562 & 0.326 & 0.729 & 0.134 & 0.356 \\
Wisconsin & 0.727 & 0.270 & 0.781 & 0.616 & 0.311 & 0.723 & 0.187 & 0.253 \\
\bottomrule
\end{tabular}%"""

new_table = """\\begin{tabular}{l|l|ccc|cc|ccc}
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
    
    if state == 'Georgia': ho = [0.749, 0.252, 0.767, 0.630, 0.528, 0.706, 0.170, 0.380]
    elif state == 'Arizona': ho = [0.695, 0.297, 0.703, 0.609, 0.449, 0.730, 0.198, 0.490]
    elif state == 'Oklahoma': ho = [0.737, 0.255, 0.747, 0.627, 0.448, 0.710, 0.147, 0.369]
    elif state == 'Wisconsin': ho = [0.727, 0.264, 0.733, 0.600, 0.439, 0.721, 0.160, 0.407]
        
    new_table += f"""
\\multirow{{4}}{{*}}{{\\textbf{{{state}}}}} 
& \\added{{MarkovChain}} & \\added{{{ga_mc['accuracy']:.3f}}} & \\added{{{ga_mc['edit_dist']:.3f}}} & \\added{{{ga_mc['bleu_score']:.3f}}} & \\added{{{ga_mc['micro_hour']:.3f}}} & \\added{{{ga_mc['macro_int']:.3f}}} & \\added{{{ga_mc['data_jsd']:.3f}}} & \\added{{{ga_mc['act_type']:.3f}}} & \\added{{{ga_mc['traj_len']:.3f}}} \\\\
& \\added{{FreqBased}} & \\added{{{ga_fb['accuracy']:.3f}}} & \\added{{{ga_fb['edit_dist']:.3f}}} & \\added{{{ga_fb['bleu_score']:.3f}}} & \\added{{{ga_fb['micro_hour']:.3f}}} & \\added{{{ga_fb['macro_int']:.3f}}} & \\added{{{ga_fb['data_jsd']:.3f}}} & \\added{{{ga_fb['act_type']:.3f}}} & \\added{{{ga_fb['traj_len']:.3f}}} \\\\
& \\added{{Rule-based}} & \\added{{{ga_rb['accuracy']:.3f}}} & \\added{{{ga_rb['edit_dist']:.3f}}} & \\added{{{ga_rb['bleu_score']:.3f}}} & \\added{{{ga_rb['micro_hour']:.3f}}} & \\added{{{ga_rb['macro_int']:.3f}}} & \\added{{{ga_rb['data_jsd']:.3f}}} & \\added{{{ga_rb['act_type']:.3f}}} & \\added{{{ga_rb['traj_len']:.3f}}} \\\\
& \\textbf{{HoMe-Llama-FT}} & {ho[0]:.3f} & {ho[1]:.3f} & {ho[2]:.3f} & {ho[3]:.3f} & {ho[4]:.3f} & {ho[5]:.3f} & {ho[6]:.3f} & {ho[7]:.3f} \\\\
\\midrule"""

new_table = new_table.rstrip('\\midrule') + "\\bottomrule\n\\end{tabular}%"
text = text.replace(old_table, new_table)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)

