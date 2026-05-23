import argparse
import os
import json
import re
from typing import Dict, List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from src.hugging_face.hf_completion_utils import get_processor_and_model
from src.llm.parse_chains import parse_reasoning_chains
from src.llm.llm_scoring import match_atomic_keys_orig, match_atomic_keys_embeddings

class BaseScorer(ABC):
    """Abstract base class for all scoring strategies."""
    @abstractmethod
    def score(self, candidate: str, ground_truth: str) -> Dict[str, float]:
        pass

class KeyPointsScorer(BaseScorer):
    """Original exact-match or embedding-based key scoring."""
    def __init__(self, model, tokenizer, sentence_transformer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.st_model = sentence_transformer

    def score(self, candidate: str, ground_truth: str) -> Dict[str, float]:
        if self.st_model:
            p, r = match_atomic_keys_embeddings(
                candidate, ground_truth, self.model, self.tokenizer, self.st_model
            )
        else:
            p, r = match_atomic_keys_orig(candidate, ground_truth, self.model, self.tokenizer)
        
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}


class EvaluationPipeline:
    def __init__(self, scorer: BaseScorer, output_path: str):
        self.scorer = scorer
        self.output_path = output_path
        self.results = []
        self.completed = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    self.results = json.load(f)
                self.completed = {
                    (item["challenge"], int(item["round"]))
                    for item in self.results
                    if "challenge" in item and "round" in item
                }
                print(f"Resuming from {output_path} with {len(self.results)} existing results.")
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
                print(f"Warning: failed to load existing results from {output_path}: {exc}")
                self.results = []
                self.completed = set()

    def run(self, challenge_path: str, pattern: str, rounds: int = 5, modulo: int = 1, remainder: int = 0):
        dirs = sorted([d for d in os.listdir(challenge_path) 
                       if os.path.isdir(os.path.join(challenge_path, d))])
        if modulo > 1:
            dirs = [d for idx, d in enumerate(dirs) if idx % modulo == remainder]

        for dir_name in dirs:
            dir_path = os.path.join(challenge_path, dir_name)
            gt_path = self._resolve_gt_path(dir_path)
            
            if not gt_path:
                continue
                
            gt_chains = parse_reasoning_chains(gt_path)
            if not gt_chains:
                print(f"Skipping {dir_name}: no ground-truth chains parsed from {gt_path}")
                continue

            for i in range(rounds):
                round_num = i + 1
                if round_num > len(gt_chains):
                    print(f"Skipping [{dir_name}] Round {round_num}: only {len(gt_chains)} ground-truth chains available")
                    continue
                if (dir_name, round_num) in self.completed:
                    continue
                cand_path = os.path.join(dir_path, f"{pattern}{round_num}.txt")
                
                if not os.path.exists(cand_path):
                    continue

                with open(cand_path, "r") as f:
                    candidate_text = f.read()

                metrics = self.scorer.score(candidate_text, gt_chains[i])
                
                result = {
                    "challenge": dir_name,
                    "round": round_num,
                    **metrics
                }
                self.results.append(result)
                self.completed.add((dir_name, round_num))
                self.save()
                print(f"[{dir_name}] Round {round_num}: {metrics}")

    @staticmethod
    def _resolve_gt_path(dir_path: str) -> str | None:
        candidates = [
            "Human_Chain_3.txt",
            "Human_Expert_3.txt",
            "Human_Expert_2.txt",
            "Human_Expert_1.txt",
        ]
        for filename in candidates:
            path = os.path.join(dir_path, filename)
            if os.path.exists(path):
                return path
        return None

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

class BipartiteLLMScorer(BaseScorer):
    """
    LLM-as-a-judge scorer that performs bipartite matching between 
    individual points in the candidate and ground truth chains.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_points(self, text: str) -> List[str]:
        """Cleans and splits reasoning chains into individual atomic points."""
        # Handles various formats: bullet points, numbered lists, or newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Remove common LLM prefixes/noise
        cleaned = [re.sub(r'^(\d+\.|\*|-)\s*', '', p).strip() for p in lines]
        return [p for p in cleaned if len(p) > 5] # Filter out tiny fragments

    def _get_llm_score(self, first_chain: str, second_point: str, metric_type: str) -> float:
        """Calls the LLM to score a single point against a reference chain."""
        prompt = f"""Provided below are two reasoning chains, the first following \"FIRST REASONING CHAIN:\" which is the ground truth
        and the second following \"SECOND REASONING CHAIN:\" which is a candidate chain. 
        These represent reasoning chains for determining the location of a photo and these chains make statements about scene attributes such as, but not limited to infrastructure, architecture, 
        vegetation and agriculture, geology, topological features, cultural clues, vehicles, and language signs. Do not restrict your grading to only these attributes and these attributes can overlap with each other.
        The first reasoning chain has multiple statement/points seperated by the "||" delimeter. 
        The second reasoning chain only contains one statement/point.
        Your task is to assign a similarity score (0–100) based **only on the SECOND CHAIN's content**, according to how well it aligns with any statement in the first chain in terms of both:
        - The **evidence** being discussed, and 
        - The **logical reasoning** or interpretation applied to that evidence.

        STRICT SCORING RULES:
        1. Only award points if there is logical evidence in the first reasoning chain that supports the second chain.
        - The second chain does not need to reflect every point in the first chain.
        - A single overlapping detail (e.g., both mention “wooden poles”) does not justify a high score, you need to consider all evidence details from the second chain.  
        - The second chain must align in **most or all** relevant evidence and reasoning dimensions of the matched point to score highly.

        2. Do NOT give any credit for:
        - Overlaps that are only geographic, thematic, or plausible but unsupported by direct evidence.  
        - If the overlap is merely “regionally plausible” or “mentions the same area,” but the specific evidence type is absent in the first chain.  
        - Thematic alignment without shared evidence != partial credit.

        3. Score 100 if the second chain has complete overlap with any of the points in the first reasoning chain, using the SAME evidence type and SAME logical reasoning.

        4. Score 0 if:
        - There is no evidence support from the first reasoning chain that supports the second chain.
        - The overlap is purely thematic (for example, both mention the Philippines but discuss different kinds of evidence with no overlap).
        - If the chain is entirely unrelated in both evidence and logic.

        5. Intermediate scores (1–99) should be given when:
        - There are evidence in the first reasoning chain (without inferring based on external knowledge) that partially support the second reasoning chain but may not have a direct overlap.

        6. You should not grade based on whether the second chain includes all the evidence or reasoning of the first chain.

        Please respond **only** with a JSON object. No explanation. Example format: {"precision": <score>}.
        FIRST REASONING CHAIN:{first_chain}
        SECOND REASONING CHAIN: {second_point}
        """

        # Using the text_to_text utility from your original script
        from src.hugging_face.hf_completion_utils import text_to_text
        response = text_to_text(prompt, self.model, self.tokenizer)
        
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return float(data.get(metric_type, 0))
            except (ValueError, KeyError):
                return 0.0
        return 0.0

    def score(self, candidate: str, ground_truth: str) -> Dict[str, float]:
        cand_points = self._extract_points(candidate)
        gt_points = self._extract_points(ground_truth)
        
        if not cand_points or not gt_points:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        gt_full_str = " || ".join(gt_points)
        cand_full_str = " || ".join(cand_points)

        # Calculate Precision: How many candidate points are in the Ground Truth?
        precisions = [self._get_llm_score(gt_full_str, p, "precision") for p in cand_points]
        avg_precision = sum(precisions) / len(precisions)

        # Calculate Recall: How many Ground Truth points are in the Candidate?
        recalls = [self._get_llm_score(cand_full_str, p, "recall") for p in gt_points]
        avg_recall = sum(recalls) / len(recalls)

        f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall) 
              if (avg_precision + avg_recall) > 0 else 0.0)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1
        }
    
class VLMJudgeScorer(BaseScorer):
    """
    A hybrid scorer that uses a VLM to assess visual correctness
    and an LLM to integrate that correctness into a final reasoning score.
    """
    def __init__(self, vlm_model, llm_model, llm_tokenizer):
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

    def score(self, candidate: str, ground_truth: str, **kwargs) -> Dict[str, float]:
        image_path = kwargs.get("image_path")
        challenge_name = kwargs.get("challenge_name", "unknown")
        round_num = kwargs.get("round_num", 0)

        if not image_path or not os.path.exists(image_path):
            print(f"Warning: No image found at {image_path}. Correctness set to 0.")
            vlm_correctness = 0.0
        else:
            # 1. Get visual correctness from VLM
            from src.vlm.correctness_score import get_correctness_score_granular
            vlm_correctness = get_correctness_score_granular(
                candidate, image_path, self.vlm_model
            )

        # 2. Use LLM to score the chain considering the VLM's visual feedback
        # Note: score_with_llm in your script writes to a file; 
        # for a clean OO approach, we ideally want it to return values.
        from src.llm.llm_scoring import score_with_llm
        
        # We pass a dummy file-like object if we want to capture output, 
        # or rely on the script's internal writing logic.
        score_with_llm(
            candidate, 
            ground_truth, 
            None, # File handle handled by pipeline save or internal script
            self.llm_model, 
            self.llm_tokenizer, 
            1.0, 
            vlm_correctness, 
            challenge_name=challenge_name, 
            round_num=round_num
        )

        return {
            "vlm_correctness": vlm_correctness,
            "status": "processed"
        }

class ScorerFactory:
    @staticmethod
    def get_scorer(mode: str, **kwargs) -> BaseScorer:
        if mode == "vlm_judge":
            return VLMJudgeScorer(
                vlm_model=kwargs.get("vlm_model"),
                llm_model=kwargs.get("llm_model"),
                llm_tokenizer=kwargs.get("llm_tokenizer")
            )
        elif mode == "key_points":
            return KeyPointsScorer(
                model=kwargs.get("llm_model"),
                tokenizer=kwargs.get("llm_tokenizer"),
                sentence_transformer=kwargs.get("st_model")
            )
        elif mode == "bipartite":
            return BipartiteLLMScorer(
                model=kwargs.get("llm_model"),
                tokenizer=kwargs.get("llm_tokenizer")
            )
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")
        
def main():
    parser = argparse.ArgumentParser(description="Extensible Reasoning Chain Evaluator")
    # Selection Argument
    parser.add_argument('--mode', choices=['key_points', 'bipartite', 'vlm_judge'], required=True,
                        help="Choose scoring logic: 'key_points', 'bipartite', or 'vlm_judge'")
    
    # Path/Config Arguments
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--vlm_model_name', help="Required if mode is vlm_judge")
    parser.add_argument('--challenge_path', required=True)
    parser.add_argument('--pattern', default="candidate_reasoning_chain_")
    parser.add_argument('--suffix', default="eval_results")
    parser.add_argument('--cache_dir', default="./cache")
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--modulo', type=int, default=1)
    parser.add_argument('--remainder', type=int, default=0)
    parser.add_argument('--emb', action='store_true', help="Only relevant for 'atomic' mode")
    
    args = parser.parse_args()

    # 1. Resource Initialization (Shared across strategies)
    print(f"--- Initializing Resources for {args.mode} mode ---")
    vlm = None
    if args.mode == "vlm_judge":
        print("Loading VLM Model...")
        from src.vllm.vllm_completion_util import get_model
        vlm = get_model(args.vlm_model_name, cache_dir=args.cache_dir)
    llm, tokenizer = get_processor_and_model(args.llm_model, args.cache_dir, device_id=0)
    
    st_model = None
    if args.mode == "key_points" and args.emb:
        st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 2. Strategy Selection via Factory
    try:
        scorer = ScorerFactory.get_scorer(
            args.mode, 
            llm_model=llm,
            llm_tokenizer=tokenizer,
            vlm_model=vlm,
            st_model=st_model
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 3. Execution
    output_file = f"vlm_scores_{args.mode}_{args.suffix}.json"
    pipeline = EvaluationPipeline(scorer, output_file)
    
    print(f"--- Starting Evaluation on {args.challenge_path} ---")
    pipeline.run(args.challenge_path, args.pattern, rounds=args.rounds, modulo=args.modulo, remainder=args.remainder)
    pipeline.save()
    
    print(f"--- Evaluation Complete. Results saved to {output_file} ---")

if __name__ == "__main__":
    main()
