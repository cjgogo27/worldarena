import argparse

def read_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('challenge_path', help="Path to the challenge file containing reasoning chains")
    parser.add_argument('hugging_face_token', help="Hugging Face token for model access")
    parser.add_argument('cache_dir', help="Cache directory for model files")
    parser.add_argument('--output_file', help="Output file for LLM judging")
    parser.add_argument('--comparison_mode', help="Comparison mode for LLM judging")
    args = parser.parse_args()
    if args.output_file:
        return args.model, args.challenge_path, args.hugging_face_token, args.cache_dir, args.output_file, args.comparison_mode
    else: 
        return args.model, args.challenge_path, args.hugging_face_token, args.cache_dir