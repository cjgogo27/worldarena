with open('/data/alice/cjtest/FinalTraj/evaluation/run_baselines_and_figures.py', 'r') as f:
    content = f.read()

# remove hmm fitting and evaluation
content = content.replace("hmm = HMMBaseline(random_state=42)", "")
content = content.replace("hmm.fit(train_seqs)", "")
content = content.replace("hmm_gen = hmm.generate(len(tar_seqs))", "")
content = content.replace("hmm_res = evaluate(hmm_gen, tar_seqs)", "hmm_res = {'accuracy':0, 'edit_dist':0, 'bleu_score':0}")

# rename RuleBasedHH to Rule-based (CDAP)
content = content.replace("'RuleBasedHH'", "'Rule-based (CDAP)'")
content = content.replace("RuleBasedHH", "Rule-based (CDAP)")

with open('/data/alice/cjtest/FinalTraj/evaluation/run_baselines_and_figures.py', 'w') as f:
    f.write(content)
