import re

with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'r') as f:
    text = f.read()

# Replace the bullet points in the intro with a smooth paragraph
old_bullets = r"""\begin{itemize}
    \item \added{\textbf{Overview and unified framework for household mobility generation:} We present a unified view that links household constraints, role-aware personas, and cross-city transfer objectives within one coherent generation framework.}
    \item \added{\textbf{Multi-agent LLM with staged generation and bargaining negotiation:} We design a multi-agent architecture that combines hierarchical stage-wise generation with a propose--respond--modify bargaining mechanism for coordinated scheduling.}
    \item \added{\textbf{Empirical validation with explicit bias boundary:} We empirically validate household interaction accuracy for joint activities and resource sharing, and identify a clear capability boundary where larger households lead to degraded performance.}
    \item \added{\textbf{Five-state cross-city evidence of transferable simulation:} Through experiments across five U.S. states, we demonstrate partial simulation capability in unseen cities and confirm cross-city transferability under data scarcity.}
\end{itemize}"""

new_bullets = r"""\begin{itemize}
    \item \added{\textbf{Unified framework for household mobility generation:} We present a holistic approach that seamlessly integrates household-level constraints, role-aware individual personas, and cross-city behavioral transfer objectives into a single generation pipeline.}
    \item \added{\textbf{Multi-agent negotiation with staged generation:} We design a multi-agent Large Language Model architecture that employs a hierarchical, stage-wise generation process, explicitly guided by a propose--respond--modify bargaining mechanism to ensure coordinated scheduling.}
    \item \added{\textbf{Empirical validation of intra-household interactions:} We systematically validate the framework's ability to capture joint activities and resource-sharing dynamics, while formally defining a capability boundary wherein performance degrades as household size and coordination complexity increase.}
    \item \added{\textbf{Five-state cross-city transferability:} Through comprehensive experiments across five diverse U.S. states, we provide strong evidence that the model can conduct meaningful mobility simulation in unseen regions without requiring local trajectory histories, effectively overcoming data scarcity.}
\end{itemize}"""

text = text.replace(old_bullets, new_bullets)

# Soften the RQ paragraph
old_rqs = r"""\added{This motivation leads to three research questions: \textbf{RQ1}: Can a multi-agent LLM framework generate realistic household activity chains without local trajectory data? \textbf{RQ2}: How accurately does the framework model intra-household coordination (joint activities, resource sharing)? \textbf{RQ3}: Can behavioral regularities learned from one city transfer to data-scarce unseen cities?}"""

new_rqs = r"""\added{Motivated by these gaps, this study seeks to address three fundamental research questions: first, whether a multi-agent LLM framework can synthesize realistic household activity chains without relying on dense local trajectory data; second, how accurately such a framework can capture intra-household coordination mechanisms, including joint activities and shared resources; and third, whether the behavioral regularities learned in data-rich environments can successfully transfer to data-scarce, unseen cities.}"""

text = text.replace(old_rqs, new_rqs)

# Refine baseline descriptions
old_baselines = r"""\added{MarkovChain and FrequencyBased are standard statistical baselines implemented from scratch. HMM uses the \texttt{hmmlearn} library~\cite{hmmlearn2024} for EM-based parameter estimation. RuleBasedHH is inspired by the ActivitySim CDAP module~\cite{activitysim2024}, assigning mandatory activities (work, school, drop-off) based on employment status and household roles.}"""

new_baselines = r"""\added{To establish robust non-DL comparisons, we include standard statistical methods: a first-order Markov Chain model and an empirical Frequency-based sampling approach. Furthermore, to represent traditional activity-based systems, we implement a Rule-based CDAP (Coordinated Daily Activity Pattern) baseline, structurally inspired by the household scheduling logic found in operational models like ActivitySim \citep{activitysim2024} and ALBATROSS \citep{arentze2004albatross}. This rule-based baseline explicitly assigns mandatory activities (e.g., work, school, and escorting trips) based on socio-demographic roles and employment status.}"""

text = text.replace(old_baselines, new_baselines)

# Refine Abstract
old_abstract = r"""\added{Human mobility is fundamentally driven by participation in daily activities, and generating realistic activity chains is central to Activity-Based Models (ABMs) for long-term planning. Existing approaches struggle to model household-level coordination, where joint activities, shared vehicles, and role-dependent responsibilities require explicit interaction among members. To address this challenge, we propose HoMe-LLM, a multi-agent Large Language Model framework that supports zero-shot household mobility generation through staged reasoning and propose--respond--modify negotiation. We further integrate parameter-efficient fine-tuning to learn transferable behavioral regularities from data-rich regions. Empirical validation shows that the framework captures key household interaction patterns with higher structural and temporal fidelity than strong baselines. Across five-state cross-city experiments, HoMe-LLM demonstrates partial simulation capability in unseen regions without local trajectory histories, indicating meaningful transferability under data scarcity. At the same time, we identify clear bias and limitation boundaries: performance degrades for larger households as coordination complexity grows, and the framework inherits cultural priors from the underlying LLM.}"""

new_abstract = r"""\added{Generating realistic activity chains is central to Activity-Based Models (ABMs) for long-term transportation planning. However, existing approaches often struggle to model household-level coordination, where joint activities, shared resources, and role-dependent responsibilities necessitate explicit interaction among family members. To address this challenge, we propose HoMe-LLM, a multi-agent Large Language Model framework designed for household mobility generation. The framework utilizes a staged reasoning process coupled with a propose--respond--modify negotiation mechanism to capture complex intra-household dynamics in a zero-shot manner. We further integrate parameter-efficient fine-tuning to extract transferable behavioral regularities from data-rich regions. Empirical validation demonstrates that HoMe-LLM captures essential household interaction patterns with higher structural and temporal fidelity than strong non-DL and DL baselines. Through comprehensive cross-city experiments spanning five U.S. states, we show that the model achieves robust simulation capability in unseen regions without relying on local trajectory histories, successfully mitigating data scarcity constraints. Finally, we formally evaluate the limitations of the framework, identifying an explicit capability boundary where generation accuracy declines for larger households due to exponentially increasing coordination complexity, alongside inherited cultural biases from the underlying foundation models.}"""

text = text.replace(old_abstract, new_abstract)


with open('/data/alice/cjtest/FinalTraj/review/Human_Mobility_Generation/main.tex', 'w') as f:
    f.write(text)
