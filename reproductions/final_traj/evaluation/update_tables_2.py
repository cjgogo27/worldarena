import re

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

# Let's fix the cross city table section:
old_cross_city = r"""California & 0.752 & 0.245 & 0.778 & 0.600 & 0.307 & 0.718 & 0.183 & 0.272 \\
Georgia   & 0.749 & 0.249 & 0.809 & 0.557 & 0.343 & 0.734 & 0.156 & 0.298 \\
Arizona   & 0.695 & 0.302 & 0.745 & 0.559 & 0.333 & 0.738 & 0.192 & 0.256 \\
Oklahoma  & 0.737 & 0.262 & 0.761 & 0.562 & 0.326 & 0.729 & 0.134 & 0.356 \\
Wisconsin & 0.727 & 0.270 & 0.781 & 0.616 & 0.311 & 0.723 & 0.187 & 0.253 \\
\added{Georgia (MarkovChain)} & \added{0.601} & \added{0.397} & \added{0.635} & \added{0.110} & \added{0.072} & \added{0.557} & \added{0.051} & \added{0.133} \\
\added{Georgia (FrequencyBased)} & \added{0.602} & \added{0.395} & \added{0.545} & \added{0.366} & \added{0.566} & \added{0.725} & \added{0.049} & \added{0.833} \\
\added{Wisconsin (MarkovChain)} & \added{0.638} & \added{0.361} & \added{0.670} & \added{0.102} & \added{0.064} & \added{0.554} & \added{0.032} & \added{0.113} \\
\added{Wisconsin (FrequencyBased)} & \added{0.638} & \added{0.358} & \added{0.579} & \added{0.367} & \added{0.551} & \added{0.724} & \added{0.030} & \added{0.833} \\
\added{Arizona (MarkovChain)} & \added{0.651} & \added{0.347} & \added{0.679} & \added{0.141} & \added{0.073} & \added{0.607} & \added{0.028} & \added{0.110} \\
\added{Arizona (FrequencyBased)} & \added{0.649} & \added{0.347} & \added{0.578} & \added{0.381} & \added{0.552} & \added{0.726} & \added{0.033} & \added{0.833} \\
\added{Oklahoma (MarkovChain)} & \added{0.598} & \added{0.400} & \added{0.632} & \added{0.200} & \added{0.081} & \added{0.627} & \added{0.060} & \added{0.115} \\
\added{Oklahoma (FrequencyBased)} & \added{0.604} & \added{0.392} & \added{0.551} & \added{0.388} & \added{0.559} & \added{0.721} & \added{0.055} & \added{0.833} \\
\bottomrule
\end{tabular}%"""

text = text.replace(old_cross_city, r"""California & 0.752 & 0.245 & 0.778 & 0.600 & 0.307 & 0.718 & 0.183 & 0.272 \\
Georgia   & 0.749 & 0.249 & 0.809 & 0.557 & 0.343 & 0.734 & 0.156 & 0.298 \\
Arizona   & 0.695 & 0.302 & 0.745 & 0.559 & 0.333 & 0.738 & 0.192 & 0.256 \\
Oklahoma  & 0.737 & 0.262 & 0.761 & 0.562 & 0.326 & 0.729 & 0.134 & 0.356 \\
Wisconsin & 0.727 & 0.270 & 0.781 & 0.616 & 0.311 & 0.723 & 0.187 & 0.253 \\
\bottomrule
\end{tabular}%""")

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)
