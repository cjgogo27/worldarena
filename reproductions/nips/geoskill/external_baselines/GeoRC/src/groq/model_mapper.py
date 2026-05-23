import argparse

def map_to_model_name(model_name):
    if model_name == "google":
        return "gemma2-9b-it"
    elif model_name == "meta":
        # return "llama-3.3-70b-versatile"
        # return "meta-llama/llama-4-maverick-17b-128e-instruct"
        return "llama-3.1-8b-instant"
    elif model_name == "qwen":
        return "qwen-qwq-32b"
    elif model_name == "deepseek":
        return "deepseek-r1-distill-llama-70b"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_model_and_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('challenge_path', help="Path to the challenge file containing reasoning chains")
    args = parser.parse_args()
    return map_to_model_name(args.model), args.challenge_path