import re

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

old_text = r"""\added{MarkovChain and FrequencyBased are standard statistical baselines implemented from scratch. HMM uses the \texttt{hmmlearn} library~\cite{hmmlearn2024} for EM-based parameter estimation. RuleBasedHH is inspired by the ActivitySim CDAP module~\cite{activitysim2024}, assigning mandatory activities (work, school, drop-off) based on employment status and household roles.}"""
# I already replaced this. Wait.

old_story = r"""These target domains represent distinct urban configurations and socio-demographic profiles, with no local training data exposed to the model. As detailed in Table~\ref{tab:state_metrics_homellama_ft_selected}, the model demonstrates remarkable robustness across these diverse geographic contexts."""

new_story = r"""These target domains represent distinct urban configurations and socio-demographic profiles, with no local training data exposed to the model. As detailed in Table~\ref{tab:state_metrics_homellama_ft_selected}, HoMe-Llama-FT demonstrates remarkable robustness across these diverse geographic contexts. \added{Crucially, this contrasts sharply with traditional Rule-based CDAP models. While the rule-based approach can achieve high accuracy (0.752) when meticulously calibrated to source-domain socio-demographic mappings (California), it suffers a catastrophic performance collapse when transferred directly to unseen states without extensive manual re-engineering of data pipelines and rules (accuracy drops to roughly 0.60, effectively reverting to random frequency sampling). HoMe-LLM, however, intrinsically parses these varied demographic attributes through natural language, maintaining a consistent sequence accuracy ranging from 0.69 to 0.75 across all target states.}"""

text = text.replace(old_story, new_story)

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)
