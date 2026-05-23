import re
import time
# from src.hugging_face.hf_completion_utils import image_and_text_to_text
from src.vllm.vllm_completion_util import image_and_text_to_text
import json
# PROMPT = """
# Given an image and a reasoning chain comprised of evidences that could lead to guessing the location of that image such as lane markings, vegetation type, soil type, traffic bollards, visible language and architectural tendencies, filter out those evidences that are not attributable to the given image.
# Output only those evidences that are directly supported by the image.
# """


PROMPT = """
Given an image and a CANDIDATE statement, determine if the candidate statement is true based on the image.
Statements about Generation coverage, the camera technology, possible guess of the location of the image should always be considered as TRUE.
Output only a SINGLE WORD "true" or "false".
Following is the CANDIDATE statement:
"""


PROMPT_GRANULAR_REASONING = """
Given an image and a CANDIDATE statement, determine if the candidate statement is true.
Statements about Generation coverage, the camera technology, possible guess of the location of the image should always be considered as TRUE.
Output the reasons for the statement being true or false without any annotations or titles along with the final answer as "true" or "false".
Following is the CANDIDATE statement:
"""

PROMPT_COMPLETE_CHAIN = """
Given an image and a series of CANDIDATE statements each on a new line, determine how many of the candidate statements are true based on the image. 
Statements about Generation coverage, the camera technology, possible country or city of the image, or any other information that cannot be confirmed from the given image should always be considered as TRUE.
Output only a SINGLE NUMBER indicating the number of true candidate statements.  Following are the CANDIDATE statements: \n
"""

PROMPT_COMPLETE_CHAIN_WITH_REASON = """
Given an image and a series of CANDIDATE statements each on a new line, determine how many of the candidate statements are true based on the image.
Statements about Generation coverage, the camera technology, possible country or city of the image, or any other information that cannot be confirmed from the given image should always be considered as TRUE.
Output the reasons for each statement being true or false without any annotations or titles. Following are the CANDIDATE statements: \n
"""

def split_chain_into_evidences(reasoning_chain):
    return [e for e in reasoning_chain.split('\n') if e.strip()]

def get_correctness_score(reasoning_chain, image_path, model):
    
    total_num_statements = len(split_chain_into_evidences(reasoning_chain))
    match = None
    retry = 0
    while not match:
        res = image_and_text_to_text(
            image_path=image_path,
            prompt=PROMPT_COMPLETE_CHAIN+f"{reasoning_chain}",
            model=model,
            # processor=processor,
            # model_name=model_name
        )
        # Extract the first integer found in the response
        match = re.search(r'\d+', str(res))
        if match:
            num_correct = int(match.group())
            print(f"Total statements: {total_num_statements}, Num correct statements -> {num_correct}")
            return num_correct / total_num_statements
        else:
            print(f"Could not extract number from response: {res}. Retrying. Attempt {retry+1}/5")
            retry += 1
        if retry >= 5:
            print("Max retries reached. Returning 1.0 correctness score.")
            return 1.0
    

def get_correctness_score_granular(reasoning_chain, image_path, model):

    prompt = """Provided below is a reasoning bullet point for determining the location of a photo. This point makes a statement about scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles, language signs and use these attributes to reason about location. 
    Your task is to determine if the bullet point is TRUE or FLASE based on the provided image. 
    TRUE means that all the scene attributes that are represented by the bullet point are present in the image. 
    Important Rules to Follow:
        - Statements guessing the location such as "This location is in <country>" or "This location is in <city>" are ALWAYS TRUE.
        - Do NOT judge it as TRUE or FALSE based on the guess of the location stated in the bullet point. For example, if the bullet point is "This vegetation is typically found in <country>", but the image is actually from a different country, it is TRUE as long as the vegetation is verifiable from the image.
        - Statements about Generation coverage and the camera technology that cannot be verified from the image should always be considered as TRUE.
        - If a bullet point clearly contradicts the image, it should be marked as FALSE. For example, if it states that there are "Yellow lane markings" but the image shows "White lane markings", this should be labelled as FALSE.
    
    OUTPUT ONLY A SINGLE WORD TRUE or FALSE.
    Following is the reasoning bullet point: \n
    """
    
    statements = split_chain_into_evidences(reasoning_chain)
    true_count = 0
    for statement in statements:
        res = image_and_text_to_text(
            image_path=image_path,
            prompt=prompt+f"{statement}",
            model=model,
            max_new_tokens=10
        )
        print(f"For statement: {statement} \n the VLM judge response: {res}")
        if "true" in res.lower():
            true_count += 1
    return float(true_count) / float(len(statements))

def get_correctness_score_granular_with_reason(reasoning_chain, image_path, model, processor, ground_truth_chain, model_name="default"):
    
    statements = split_chain_into_evidences(reasoning_chain)
    true_count = 0
    for statement in statements:
        res = image_and_text_to_text(
            image_path=image_path,
            prompt=PROMPT_GRANULAR_REASONING+f"\n{statement}",
            model=model,
            processor=processor,
            model_name=model_name,
            max_new_tokens=1024
        )
        print(f"For statement: {statement} ; the VLM judge response: {res}")
        if "true" in res.lower():
            true_count += 1
    return float(true_count) / float(len(statements))

def debug_correctness_score(reasoning_chain, image_path, model, processor, ground_truth_chain):
    
    res = image_and_text_to_text(
        image_path=image_path,
        prompt=PROMPT_COMPLETE_CHAIN_WITH_REASON+f"{reasoning_chain}",
        model=model,
        processor=processor,
        max_new_tokens=1024
    )
    return res

def filter_incorrect_evidences(reasoning_chain, image_path, model, processor):
    evidences = split_chain_into_evidences(reasoning_chain)
    filtered_evidences = []
    for i, evidence in enumerate(evidences):
        start_time = time.time()
        res = image_and_text_to_text(
            image_path=image_path,
            prompt=PROMPT+f"\n{evidence}",
            model=model,
            processor=processor
        )
        end_time = time.time()
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print(f"Evidence {i+1}: {evidence} -> {res}")
        if "true" in res.lower():
            filtered_evidences.append(evidence)
    return "\n".join(filtered_evidences)