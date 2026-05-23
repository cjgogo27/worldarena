import re
import json

with open('/data/alice/cjtest/FinalTraj/evaluation/baseline_results.json', 'r') as f:
    res = json.load(f)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

# Replace the content of Table 5
table5_old = r"""\begin{tabular}{l|ccc|cc|ccc}
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

# It seems that my previous script replaced it differently. Let's find exactly what's there now.
