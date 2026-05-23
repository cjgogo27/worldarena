import os
import argparse
from abc import ABC, abstractmethod
from src.hugging_face.hf_completion_utils import get_processor_and_model, get_pipeline
from src.hugging_face.hf_completion_utils import image_and_text_to_text, image_and_text_to_text_using_pipeline
from src.gemini.model_util import image_and_text_to_text as gemini_text_gen
from src.azure.model_util import image_and_text_to_text as azure_text_gen

class ReasoningChainsGenerator(ABC):
    """Abstract Base Class for all Vision-Language model types."""
    def __init__(self, model_name, challenge_path):
        self.model_name = model_name
        self.challenge_path = challenge_path
        self.main_prompt = """
        You are a professional GeoGuessr player. You are an expert at finding the location of photographs. 
        Given an image, guess its location in terms of country and city. Also identify the specific pieces of evidence that lead you to this decision such as lane markings, vegetation type, soil type, traffic bollards, visible language, architectural tendencies, and any other clues you notice. 
        Give a detailed explanation of which pieces of geographic evidence you used to make your location estimate. 
        """
    @abstractmethod
    def image_and_text_to_text(self, image_path):
        """Must be implemented by subclasses."""
        pass
    def get_challenge_name(self, path):
        return os.path.basename(os.path.normpath(path))
    
    def generate_reasoning_chains(self, challenge_dirs_filter=None):
        """Unified execution logic for all models."""
        # Handle single directory or parent directory logic
        dirs = sorted(os.listdir(self.challenge_path)) if os.path.isdir(self.challenge_path) else [self.challenge_path]
        
        for item in dirs:
            dir_path = os.path.join(self.challenge_path, item) if os.path.isdir(self.challenge_path) else item
            if not os.path.isdir(dir_path): continue
            
            challenge_name = self.get_challenge_name(dir_path)
            if challenge_dirs_filter and challenge_name not in challenge_dirs_filter:
                continue

            print(f"Testing {self.model_name} on: {challenge_name}")
            for i in range(1, 6):
                output_file = os.path.join(dir_path, f"candidate_reasoning_chain_{self.model_name}_{i}.txt")
                if os.path.exists(output_file): continue

                image_path = os.path.join(dir_path, f"{challenge_name}_{i}.png")
                reasoning = self.image_and_text_to_text(image_path)
                
                with open(output_file, "w") as f:
                    f.write(reasoning)

class HuggingFaceGenerator(ReasoningChainsGenerator):
    def __init__(self, model_name, challenge_path, hf_token, cache_dir):
        super().__init__(model_name, challenge_path)
        os.environ["HF_TOKEN"] = hf_token
        if model_name == "google":
            self.pipeline = get_pipeline(model_name, cache_dir=cache_dir)
            self.use_pipeline = True
        else:
            self.model, self.processor = get_processor_and_model(model_name, cache_dir)
            self.use_pipeline = False

    def image_and_text_to_text(self, image_path):
        if self.use_pipeline:
            return image_and_text_to_text_using_pipeline(image_path, self.main_prompt, self.pipeline, max_new_tokens=512)
        return image_and_text_to_text(image_path, self.main_prompt, self.model, self.processor, max_new_tokens=512)
    


class GeminiApiGenerator(ReasoningChainsGenerator):
    def __init__(self, model_name, challenge_path, api_key):
        super().__init__(model_name, challenge_path)
        os.environ["GEMINI_API_KEY"] = api_key

    def image_and_text_to_text(self, image_path):
        return gemini_text_gen(image_path, self.main_prompt, self.model_name)

class AzureApiGenerator(ReasoningChainsGenerator):
    def __init__(self, model_name, challenge_path, api_key, endpoint, api_version):
        super().__init__(model_name, challenge_path)
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version

    def image_and_text_to_text(self, image_path):
        return azure_text_gen(image_path, self.main_prompt, self.endpoint, self.api_key, self.api_version, self.model_name)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['hf', 'gemini', 'azure'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--key', help="API Key or HF Token")
    parser.add_argument('--cache', default="./cache")
    parser.add_argument('--endpoint', help="API endpoint for Azure")
    parser.add_argument('--api_version', help="API endpoint version for Azure")
    args = parser.parse_args()

    if args.type == 'hf':
        generator = HuggingFaceGenerator(args.model, args.path, args.key, args.cache)
    elif args.type == 'gemini':
        generator = GeminiApiGenerator(args.model, args.path, args.key)
    elif args.type == 'azure':
        generator = AzureApiGenerator(args.model, args.path, args.key, args.endpoint, args.api_version)
    
    generator.generate_reasoning_chains()

if __name__ == "__main__":
    main()