import src.hugging_face.hf_completion_utils as completion_utils
import re
import json
from sentence_transformers import util

try:
    import src.vllm.vllm_completion_util as vllm_completion_utils
except ModuleNotFoundError:
    vllm_completion_utils = None

"""
You are an expert at summarizing a statement.
            \nTASK:
            Summarize the given STATEMENT into less than 3 pipe delimited (|) key clauses without any reasoning and annotations.
            \nDEFINITIONS:
            The STATEMENT describes one or more scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that are used to reason about the location of an image.
            The LIST of key clauses should summarize the idea of the STATEMENT.
            \nRULES:
            1. Countries, cities, continents, regions or any other location names should NOT be included in the key clauses. For example, "spain", "barcelona", "asia", "southeast asia" are NOT allowed.
            2. Use the words from the statement only. Do not rephrase or change the words.
            3. Key clauses should include a very brief one or two words description for common objects and scene attributes. For example, "deciduous trees", "blue car" and "open rolling hills" are descriptive whereas "trees", "car" and "hills" are not.
            4. Adjectives that describe the scene attribute or object should be in the same clause and not be separated. For example, "red sign" is valid whereas "red" is not. 
            5. Negations should be represented in the same phrase. For example, if the statement says, "not russian or acryllic characters" then "not russian characters" and "not acryllic characters" are expected.
            6. Verbs and adverbs alone are invalid clauses. For example, "obscured by trees" is valid but "obscured by" is invalid.
            7. Positions in the image are not important and can be excluded from the key clause. For example, for "black car at the bottom of the image", the summary key clause will be "black car". 
            8. After applying all these rules, if the STATEMENT does not have any important clauses, return <empty> as the response.
            \nINPUT:
            \nSTATEMENT: {point}
            \nOUTPUT:
            <empty> if no key clauses were found,
            <clause 1> if STATEMENT can be summarized into a single clause,
            <clause 1> | <clause 2> if STATEMENT can be summarized into 2 clauses,
            <clause 1> | <clause 2> | <clause 3> if STATEMENT can be summarized into 3 clauses
"""
def score_with_llm(candidate_reasoning_chain, ground_truth_reasoning_chain, score_file, model, tokenizer, gt_accuracy=100.0, candidate_accuracy = 100.0, challenge_name="default_challenge_id", round_num=-1):
    
    score_prompt = """Provided below are two reasoning chains, the first following \"GROUND TRUTH REASONING CHAIN:\" and the second following \"CANDIDATE REASONING CHAIN:\". 
    These represent reasoning bullet points for determining the location of a photo and these points make statements about scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles, language signs and use these attributes to reason about location. 
    The ground truth reasoning chain is written by an expert who is always correct. 
    
    Your task is to compare the candidate bullet points to the ground truth and score it, out of 100, in terms of precision and recall. 
    Precision measures similarity of the candidate points to the ground truth reasoning chain. 
    Recall measures similarity of the ground truth bullet points to the candidate reasoning chain.

    Similarity is defined as how much the statements logically supports, reinforces, and overlaps with in terms of geographical evidence.
    A score of 100 means that the candidate reasoning chain is a direct paraphrasis of the points in the ground truth reasoning chain.
        
    Important Rules to Follow:
        - Bullet points are the core unit that goes into the numerator and denominator of the recall and precision calculations. E.g. if there are 7 non-concluding, image-evidence bullet points, that will be the denominator. This is true even if 3 of the bullets are about road infrastructure.
        - Each bullet point has equal amount of “support” from the other chain and contributes equally to the score out of 100.
        - Bullet points can be supported by multiple bullet points in the other chain. It does not need to be 1 to 1 matching. It can be 1 to many.
        - Bullet points that are restating previous evidence should be disregarded (affecting neither the numerator or denominator). This goes beyond explicit “conclusion” bullets. Sometimes there are multiple conclusion-like bullet points. Sometimes there are bullet points that elaborate on the previous bullet points without pointing to new image evidence. Disregard these.
        - “White outer lines typical of Africa” and “White outer lines typical of Middle East” are in agreement. We are judging the image properties, not the correctness of the geographic support of those properties.
        - Only award points if the same type of evidence appears in both chains (example, both mention architecture, or both mention plant species).
        - Do NOT award points for general overlap in country, theme, or plausibility.
        - If the candidate chain uses a different evidence type (example, it discusses architecture, but the ground truth chain only talks about vegetation), the precision score must be 0 for this bullet point.
        - Do not infer connections that are not explicitly stated.
        - Do not award points simply because both chains arrive at the same conclusion (example, same country) if the evidence types are unrelated or not logically connected.
        - The reasoning chains should be logically consistent. If candidate reasoning chain introduces evidence that contradicts or is irrelevant to the other, score 0 for precision.
        
    OUTPUT ONLY A JSON BLOB representing these two values as follows: { \"precision\": , \"recall\": }
    
    GROUND TRUTH REASONING CHAIN:

    """ + ground_truth_reasoning_chain + """
    
    CANDIDATE REASONING CHAIN:

    """ + candidate_reasoning_chain
    
    score_blob = completion_utils.text_to_text(score_prompt, model, tokenizer)
    json_match = re.search(r'\{(.*?)\}', score_blob, re.DOTALL)
    if json_match:
        extracted_json = json_match.group()
        data = json.loads(extracted_json)
        precision = data.get("precision", 0)
        precision *= (candidate_accuracy/100.0)
        data["precision"] = precision
        recall = data.get("recall", 0)
        recall *= (gt_accuracy/100.0)
        data["recall"] = recall
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        data["f1"] = f1
        data["challenge"] = challenge_name
        data["round"] = round_num
        data["correctness"] = candidate_accuracy
        extracted_json = json.dumps(data)
        score_file.write(extracted_json + ",\n")
        score_file.flush()
        print(extracted_json)
    else:
        print("No JSON score output found in completion: "+score_blob)

def split_chain(reasoning_chain):
    return [e for e in reasoning_chain.split('\n') if e.strip()]

# 8. After applying all these rules, if the STATEMENT does not have any important clauses, return <empty> as the response.
def parse_into_keys(chain, model, tokenizer=None, list_str=False, use_vllm=False):
    if use_vllm and vllm_completion_utils is None:
        raise RuntimeError("vLLM utilities are unavailable in this checkout")
    prompt = """You are an expert at summarizing a statement.
    \nTASK:
    Summarize the given STATEMENT into a pipe delimited (|) LIST of key clauses without any reasoning and annotations.
    \nDEFINITIONS:
    The STATEMENT describes one or more scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that are used to reason about the location of an image.
    The LIST of key clauses should summarize the idea of the STATEMENT.
    \nRULES:
    1. Countries, cities, continents, regions or any other location names should NOT be included in the key clauses. For example, "spain", "barcelona", "asia", "southeast asia" are NOT allowed.
    2. Use many descriptive words in the key clause but from the statement only. Do not rephrase or change the words. For example, for the statement "The camera angle of the image is lower (lowcam)", "lower camera angle" is the correct key clause.
    3. All words pertaining to the same object or attribute should be in the same key clause. For example, "short bollard with vertical reflector and black cap" are in a single key clause as they describe the same object, bollard.
    4. Key clauses should include the description for common objects and scene attributes. For example, "deciduous trees", "blue car" and "open rolling hills" are descriptive whereas "trees", "car" and "hills" are not.
    5. Negations should be represented in the same phrase. For example, if the statement says, "not russian or acryllic characters" then "not russian characters" and "not acryllic characters" are expected.
    6. Verbs and adverbs alone are invalid clauses. For example, "obscured by trees" is valid but "obscured by" is invalid.
    7. Positions in the image are not important and can be excluded from the key clause. For example, for "black car at the bottom of the image", the summary key clause will be "black car". 
    \nINPUT:
    \nSTATEMENT: 
    """
    if list_str:
        keys = []
    else:
        keys = {}
    num_statements = 0
    for point in split_chain(chain):
        
        

        if use_vllm:
            res = vllm_completion_utils.text_to_text_qwen_30b(prompt+point+"\nOUTPUT: LIST:", model, 1024, tokenizer)
        else:    
            res = completion_utils.text_to_text(prompt+point+"\nOUTPUT: LIST:", model, tokenizer)
        
        complete_point_keys = [k.strip().lower() for k in res.split('LIST:')[-1].split('|') if k.strip()]

            # print(f"for {point}\n the result from model was {res} and keys parsed were: {curr_keys}")
            
        # if len(complete_point_keys) > 0 and complete_point_keys[0] != '<empty>':
        if len(complete_point_keys) > 0:
            num_statements += 1
            points_per_key = 1.0/len(complete_point_keys)
            if list_str:
                keys.append(complete_point_keys)
            else:
                for k in complete_point_keys:
                    if k in keys:
                        keys[k] += points_per_key
                    else:
                        keys[k] = points_per_key
    if not list_str:
        keys.update((key, value/num_statements) for key, value in keys.items())
    return keys

def cal_key_matches_by_llm(candidate_chain, ground_truth_chain, model, tokenizer):
    gt_keys = parse_into_keys(ground_truth_chain, model, tokenizer, True)
    candidate_keys = parse_into_keys(candidate_chain, model, tokenizer, True)
    
    precision_values = []
    for point in candidate_keys:
        local_precision_values = []
        for gt_s in gt_keys:
            match_prompt = f"""You are an expert evaluator of matching clauses in English language.
            \nTASK:
            Without any reasoning and annotations determine how many of the KEY clauses in LIST 1 are similar to the KEY clauses in LIST 2.
            \nDEFINITIONS:
            KEY is a clause about a scene attribute of an image such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that are used to reason about the location of an image.
            Both LIST 1 and LIST 2 are lists of KEY clauses.
            \nRULES:
            1. Words or phrases that refer to the same object as the key are similar. For example, "white truck" is similar to "truck". But "corn field" is not similar to "agricultural field".
            2. Contradicting keys are not similar. For example, "completely flat" is not similar to "not completely flat".
            3. Keys that are referring to the same scene attribute are similar. For example, "tropical environment" is similar to "tropical climate". Key "green shrubs" is similar to "green bushes".
            4. Keys are similar if some additionally describing words are missing, as long as the overall key retains the same meaning about the scene attribute as the ground truth key. For example, key "short bollards with a vertical reflector and a black cap on top" is similar to "white bollards with the black cap" because both keys consider bollards. Similarly, the key "black car" is similar to "black colored google car" because both the keys refer to a black car.
            \n
            \nINPUTS:
            LIST 1: "{gt_s}"
            LIST 2: "{point}"
            \nOUTPUT: a NUMBER less than {len(gt_s)+1}, indicating the number of keys found similar in LIST 1 when compared to LIST 2.
            """
            match_res = completion_utils.text_to_text(match_prompt, model, tokenizer)
            print("Match response:", match_res)
            match = re.search(r'\d+', str(match_res))
            if match:
                num_correct = int(match.group())
                score = num_correct*1.0 / len(gt_s)

                print(f"Keys Num found: {score}")
                local_precision_values.append(score)
            else:
                print("Could not extract number from response:", match_res)
                local_precision_values.append(0.0)
        precision_values.append(max(local_precision_values))
    return sum(precision_values)/len(precision_values)

def cal_key_matches_by_sentence_transformer(candidate_chain, ground_truth_chain, model, tokenizer, sentence_transformer):
    gt_keys = parse_into_keys(ground_truth_chain, model, tokenizer)
    candidate_keys = parse_into_keys(candidate_chain, model, tokenizer)
    
    gt_embeddings = [(sentence_transformer.encode(k, convert_to_tensor=True),v,k) for k,v in gt_keys.items()]
    candidate_embeddings = [(sentence_transformer.encode(k, convert_to_tensor=True),v,k) for k,v in candidate_keys.items()]
    
    precision = 0.0
    recall = 0.0
    score_added_cand = set()
    for i,eg in enumerate(gt_embeddings):
        gt_em,gt_score,gt_key = eg
        gt_matched = False
        for j,ec in enumerate(candidate_embeddings):
            
            cand_em,cand_score,cand_key = ec
            sim = util.pytorch_cos_sim(gt_em, cand_em)
            print(f"sim between {gt_key};{cand_key} = {sim}")
            if sim >=0.45:
                if not cand_key in score_added_cand:
                    precision += cand_score
                    score_added_cand.add(cand_key)
                gt_matched = True
        if gt_matched:
            recall += gt_score
    return precision, recall

def encode_keys(chain_matrix, sentence_transformer):
    embeddings = []
    for i in range(len(chain_matrix)):
        line_emb = []
        for j in range(len(chain_matrix[i])):
            curr_key = chain_matrix[i][j]
            line_emb.append((sentence_transformer.encode(curr_key, convert_to_tensor=True), curr_key))
        embeddings.append(line_emb)
    return embeddings

def compute_line_similarity(gt_line, cand_line):
    num_matched_gt,num_matched_cand = set(),set()
    for gt_key in gt_line:
        for cand_key in cand_line:
            sim = util.pytorch_cos_sim(gt_key[0], cand_key[0])
            
            if sim >=0.40:
                print(f"sim between {gt_key[1]};{cand_key[1]} = {sim}")
                num_matched_gt.add(gt_key[0])
                num_matched_cand.add(cand_key[0])

    recall = len(num_matched_gt)*1.0/len(gt_line)
    precision = len(num_matched_cand)*1.0/len(cand_line)
    return precision, recall

def compute_line_similarity_ramp(gt_line, cand_line, LOWER_BOUND, UPPER_BOUND):
    matched_gt = {k[1]:0.0 for k in gt_line}
    matched_cand = {k[1]:0.0 for k in cand_line}

    for gt_key in gt_line:
        for cand_key in cand_line:
            sim = util.pytorch_cos_sim(gt_key[0], cand_key[0]).item()
            
            if sim <= LOWER_BOUND:
                sim = 0.0
            elif sim >=UPPER_BOUND:
                sim = 1.0
            else:
                sim = (sim - LOWER_BOUND)/(UPPER_BOUND - LOWER_BOUND)
            print(f"sim between {gt_key[1]};{cand_key[1]} = {sim}")
            if sim > matched_gt[gt_key[1]]:
                matched_gt[gt_key[1]] = sim
            if sim > matched_cand[cand_key[1]]:
                matched_cand[cand_key[1]] = sim

    recall = sum(matched_gt.values())/len(matched_gt)
    precision = sum(matched_cand.values())/len(matched_cand)
    return precision, recall

def ramp_score(candidate_chain, ground_truth_chain, model, tokenizer, sentence_transformer, LOWER_BOUND, UPPER_BOUND):


    gt_keys_map = parse_into_keys(ground_truth_chain, model, tokenizer, False)
    candidate_keys_map = parse_into_keys(candidate_chain, model, tokenizer, False)

    def map_with_emb(map):
        new_map = {}
        for k,v in map.items():
            new_map[k] = (sentence_transformer.encode(k, convert_to_tensor=True), v)
        return new_map
    gt_keys_map = map_with_emb(gt_keys_map)
    candidate_keys_map = map_with_emb(candidate_keys_map)

    gt_keys_sim_score = {k:0.0 for k in gt_keys_map.keys()}
    cand_keys_sim_score = {k:0.0 for k in candidate_keys_map.keys()}
    for gt_key,(gt_key_emb,_) in gt_keys_map.items():
        for cand_key,(cand_key_emb,_) in candidate_keys_map.items():
            sim = util.pytorch_cos_sim(gt_key_emb, cand_key_emb).item()
            if sim <= LOWER_BOUND:
                sim = 0.0
            elif sim >=UPPER_BOUND:
                sim = 1.0
            else:
                sim = (sim - LOWER_BOUND)/(UPPER_BOUND - LOWER_BOUND)
            if sim>gt_keys_sim_score[gt_key]:
                # print(f"Found greater similarity between {gt_key}; {cand_key} = {sim}")
                gt_keys_sim_score[gt_key] = sim
            if sim>cand_keys_sim_score[cand_key]:
                cand_keys_sim_score[cand_key] = sim
    def sum_score(weights_map,score_map):
        score = 0.0
        for k,(_,w) in weights_map.items():
            score += w * score_map[k]
        return score
    precision = sum_score(candidate_keys_map, cand_keys_sim_score)
    recall = sum_score(gt_keys_map, gt_keys_sim_score)
    return precision,recall

def key_to_key_matches_by_llm(key1, key2, model, tokenizer):
    match_prompt = f"""You are an expert evaluator of checking similarity of two clauses in English language.
            \nTASK:
            Without any reasoning and annotations determine if the scene attributes in the KEY1 and KEY2 are similar in meaning based on the provided set of RULES and return a single answer TRUE or FALSE.
            \nDEFINITIONS:
            The clause in KEY1 and KEY2 refers to a scene attribute of an image such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that are used to reason about the location of an image.
            
            \nRULES:
            1. KEY1 and KEY2 are similar if they refer to the same object. For example, TRUE should be returned for keys, "wooden electricity poles" and "ooden utility pole". Return FALSE for keys "corn field" and "agricultural field in the distance" as corn field is not equal to "agricultural field".
            2. Contradicting keys are FALSE. For example, "completely flat" is not supported by "not completely flat".
            3. Return TRUE for keys that are similar in meaning when placed in the context of the scene attributes. For example, return TRUE for keys "tropical environment" and "tropical climate". TRUE for keys "green shrubs" and "green bushes".
            4. Keys are similar even if some additionally describing words are missing, as long as the overall statement retains the same meaning about the scene attribute. For example, return TRUE for keys "short bollards with a vertical reflector and a black cap on top" and "white bollards with the black cap" because both keys refer to bollards. Similarly, the keys "black car" and "black colored google car at the bottom of the image" because both the keys refer to the black car.
            \n
            \nINPUTS:
            KEY1: "{key1}"
            KEY2: "{key2}"
            \nOUTPUT: TRUE if the KEY1 is similar to KEY2 and FALSE otherwise.
            """
    match_res = completion_utils.text_to_text(match_prompt, model, tokenizer)
    # print(f"eval result of key match between {key1} and {key2}: {match_res}")
    if "true" in match_res.lower():
        return True
    return False

def llm_key_scoring(candidate_chain, ground_truth_chain, model, tokenizer):

    gt_keys_map = parse_into_keys(ground_truth_chain, model, tokenizer, False)
    print(f"Ground truth keys = {gt_keys_map}")
    candidate_keys_map = parse_into_keys(candidate_chain, model, tokenizer, False)
    print(f"Candidate keys = {candidate_keys_map}")

    gt_keys_sim_score = {k:0.0 for k in gt_keys_map.keys()}
    cand_keys_sim_score = {k:0.0 for k in candidate_keys_map.keys()}
    for gt_key,_ in gt_keys_map.items():
        for cand_key,_ in candidate_keys_map.items():
            ans = key_to_key_matches_by_llm(gt_key, cand_key, model, tokenizer)
            if ans:
                gt_keys_sim_score[gt_key] = 1.0
                cand_keys_sim_score[cand_key] = 1.0
                
    def sum_score(weights_map,score_map):
        score = 0.0
        for k,w in weights_map.items():
            score += w * score_map[k]
        return score
    def print_keys_not_matched(score_map, side):
        for k,v in score_map.items():
            if v == 0.0:
                print(f"Computing {side} key not matched = {k}")
    print_keys_not_matched(cand_keys_sim_score, "precision")
    print_keys_not_matched(gt_keys_sim_score, "recall")
    precision = sum_score(candidate_keys_map, cand_keys_sim_score)
    recall = sum_score(gt_keys_map, gt_keys_sim_score)
    return precision,recall

def cal_key_matches_by_sentence_transformer_max_overlap(candidate_chain, ground_truth_chain, model, tokenizer, sentence_transformer, lt, ut):
    gt_keys = parse_into_keys(ground_truth_chain, model, tokenizer, True)
    candidate_keys = parse_into_keys(candidate_chain, model, tokenizer, True)
    
    gt_embeddings = encode_keys(gt_keys, sentence_transformer)
    candidate_embeddings = encode_keys(candidate_keys, sentence_transformer)
    
    precision = []
    recall = []
    for cand_line in candidate_embeddings:
        local_precision, local_recall = [], []
        for gt_line in gt_embeddings:
            p, r = compute_line_similarity_ramp(gt_line, cand_line, lt, ut)
            # print(f"local precision and recall between {cand_line};{gt_line}={p};{r}")
            local_precision.append(p)
            local_recall.append(r)
        precision.append(max(local_precision))
        recall.append(max(local_recall))
    precision_val, recall_val = 0.0, 0.0
    if len(precision) > 0:
        precision_val = sum(precision)/len(precision)
    if len(recall) > 0:
        recall_val = sum(recall)/len(recall)
    return precision_val, recall_val

def cal_key_matches(chain1, chain2, model, tokenizer, use_vllm=False):
    if use_vllm and vllm_completion_utils is None:
        raise RuntimeError("vLLM utilities are unavailable in this checkout")
    list2 = parse_into_keys(chain2, model, tokenizer, use_vllm=use_vllm)
    list1 = split_chain(chain1)
    num_keys = len(list2)
    if num_keys == 0:
        return 0.0
    true_positives = set()
    score = 0.0
    for line in list1:
        for key, val in list2.items():
            if key in true_positives:
                continue
            # match_prompt = f"""From the below 2 lists, LIST 1 and LIST 2, of keys that are words or phrases separated by a comma, your task is to return those keys from LIST 2 that are found similar in meaning to or are supported by the keys in LIST 1.
            # Follow the below rules:
            # - Keys are similar if they refer to the same object or concept. For example, 2 keys in ["car", "tree"] are similar to the 2 keys in ["vehicle", "vegetation"] because, "tree" and "vegetation", "car" and "vehicle" are similar, but "car" and "tree" are not similar.
            # - A key in the CANDIDATE LIST can be similar to more than one key in the ground truth list. For example, the 2 keys in ["deciduous", "trees"] are similar to the 1 key in ["deciduous trees"].
            # - Beware of contrasting or contradicting keys. "completely flat" does not match with "not completely flat".
            # - Keys can be considered as matched even if some words are missing or are different and as long as the overall key retains the same meaning without the absent words or with the different word. For example, "narrow paved road" matches "paved road", "black colored google car" matches "black google car", "tropical environment" matches "tropical climate".
            # RETURN a comma separated LIST OF VALUES FROM LIST 2, indicating the keys that matched from LIST 2 without any reasoning or any other annotations. Your answer should only have the keys from LIST 2 and nothing else is allowed. \n
            # Following are the two lists:\n
            # LIST 1: {",".join(stmt)}\n
            # LIST 2: {",".join(list2.keys())}\n
            # Begin your answer with MATCHED LIST:
            # """
            match_prompt = f"""You are an expert evaluator of checking reference of clauses in a statement in English language.
            \nTASK:
            Without any reasoning and annotations determine if the scene attribute in the KEY is referenced by the STATEMENT.
            \nDEFINITIONS:
            The KEY is a clause about a scene attribute of an image such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that are used to reason about the location of an image.
            The STATEMENT describes one or more scene attributes.
            \nRULES:
            1. Words or phrases in the statement that refer to the same object as the key are TRUE. For example, TRUE should be returned for key, "white truck" and statement "there is a truck in the image". Return FALSE for key "corn field" and statement "there is an agricultural field in the distance" as corn field is not equal to "agricultural field".
            2. Contradicting key and statement are FALSE. For example, "completely flat" is not supported by "the landscape is not completely flat".
            3. Return TRUE for keys that are similar in meaning when placed in the context of the statement. For example, key "tropical environment" is present in the stament "the setting is in a tropical climate". Key "green shrubs" is present in "green bushes along the edges".
            4. Keys are present if some additionally describing words are missing, as long as the overall statement retains the same meaning about the scene attribute as the key. For example, key "short bollards with a vertical reflector and a black cap on top" is present in statement "The white bollards with the black cap are commonly found in Austria" because the key and statement both consider bollards. Similarly, the key "black car" is present in statement "there is a black colored google car at the bottom of the image" because both the key and statement refer to a black car.
            \n
            \nINPUTS:
            KEY: "{key}"
            STATEMENT: "{line}"
            \nOUTPUT: TRUE if the KEY is present in the STATEMENT and FALSE otherwise.
            """
            if use_vllm:
                match_res = vllm_completion_utils.text_to_text_qwen_30b(match_prompt, model, 10, tokenizer)
            else:
                match_res = completion_utils.text_to_text(match_prompt, model, tokenizer, 10)
            print(f"eval result of key match between {key} and {line}: {match_res}")
            if "true" in match_res.lower():
                true_positives.add(key)
                score += val

    print(f"match results at the end of loop,\n {true_positives};\n all keys were: {list2};\n keys not matched are: ")
    for k in list2.keys():
        if k not in true_positives:
            print(k)
    print("\n")
    return score
    
def match_atomic_keys_embeddings(candidate_chain, ground_truth_chain, model, tokenizer, sentence_transformer, lt=0.40, ut=0.55):
    precision, recall = ramp_score(candidate_chain, ground_truth_chain, model, tokenizer, sentence_transformer, lt, ut)
    print(f"Overall Precision: {precision}, Recall: {recall}")
    return precision, recall

def match_atomic_keys(candidate_chain, ground_truth_chain, model, tokenizer):
    precision, recall = llm_key_scoring(candidate_chain, ground_truth_chain, model, tokenizer)
    print(f"Overall Precision: {precision}, Recall: {recall}")
    return precision, recall

def match_atomic_keys_orig(candidate_chain, ground_truth_chain, model, tokenizer):
    prompt = """Provided below is a statement describing one or more scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles and language signs that is used to reason about location.
    Your task is to only output a comma separated list of words or phrases from the statement that most accurately summarize the statement.
    Following is the statement: \n
    """
    gt_keys = []
    for gt in split_chain(ground_truth_chain):
        res = completion_utils.text_to_text(prompt+gt+"\nBegin your answer with LIST:", model, tokenizer)
        curr_keys = [k.strip().lower() for k in res.split('LIST:')[-1].split(',') if k.strip()]
        if len(curr_keys) > 0:
            gt_keys.append(curr_keys)
    print("Ground truth keys:", gt_keys)
    
    precision_values, recall_values  = [], []
    
    for c in split_chain(candidate_chain):
        res = completion_utils.text_to_text(prompt+c+"\nBegin your answer with LIST:", model, tokenizer)
        candidate_keys = [k.strip().lower() for k in res.split('LIST:')[-1].split(',') if k.strip()]
        print("Candidate keys:", candidate_keys)
        if len(candidate_keys) == 0:
            continue
        local_precision_values, local_recall_values = [], []
        for gt_s in gt_keys:
            match_prompt = f"""From the below 2 lists, CANDIDATE LIST and GROUND TRUTH LIST, of keys that are words or phrases separated by a comma, for each list your task is to count how many of the keys are found similar in meaning to the keys in the other list.
            Follow the below rules:
            - Keys are similar if they refer to the same object or concept. For example, 2 keys in ["car", "tree"] are similar to the 2 keys in ["vehicle", "vegetation"] because, "tree" and "vegetation", "car" and "vehicle" are similar, but "car" and "tree" are not similar.
            - A key in the CANDIDATE LIST can be similar to more than one key in the ground truth list. For example, the 2 keys in ["deciduous", "trees"] are similar to the 1 key in ["deciduous trees"].
            - Beware of contrasting or contradicting keys. "completely flat" does not match with "not completely flat".
            RETURN TWO NUMBERS, a CANDIDATE NUMBER less than {str(len(candidate_keys)+1)}, and a GROUND TRUTH NUMBER less than {str(len(gt_s)+1)}, indicating how many of the keys match or are similar in each list without any reasoning or any other annotations.\n
            Following are the two lists:\n
            CANDIDATE LIST: {candidate_keys}\n
            GROUND TRUTH LIST: {gt_s}\n
            Your answers, CANDIDATE NUMBER: , GROUND TRUTH NUMBER:
            """
            match_res = completion_utils.text_to_text(match_prompt, model, tokenizer)
            print("Match response:", match_res)
            candidate_match = match_res.split('CANDIDATE NUMBER:')[-1].split(',')[0]
            candidate_match = re.search(r'\d+', str(candidate_match))
            ground_truth_match = match_res.split('GROUND TRUTH NUMBER:')[-1]
            ground_truth_match = re.search(r'\d+', str(ground_truth_match))
            if candidate_match and ground_truth_match:
                num_correct_candidates = int(candidate_match.group())
                num_correct_ground_truth = int(ground_truth_match.group())
                precision = num_correct_candidates*1.0 / len(candidate_keys)
                recall = num_correct_ground_truth*1.0 / len(gt_s)

                print(f"Keys Precision: {precision}, Recall: {recall}")
                local_precision_values.append(precision)
                local_recall_values.append(recall)
            else:
                print("Could not extract number from response:", match_res)
                local_precision_values.append(0.0)
                local_recall_values.append(0.0)
        precision_values.append(max(local_precision_values))
        recall_values.append(max(local_recall_values))
    return sum(precision_values)/len(precision_values), sum(recall_values)/len(recall_values)
