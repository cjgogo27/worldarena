# GeoRC

## Installation

### Prerequisites

1. **Install Miniforge** (lightweight conda distribution):
   
   Visit [Miniforge GitHub](https://github.com/conda-forge/miniforge) and download the appropriate installer for your OS (macOS, Linux, or Windows).
   
   After downloading, run the installer:
   ```bash
   bash Miniforge3-*.sh
   ```
   
   Follow the prompts and accept the default settings.

2. **Create and activate a conda environment**:
   ```bash
   conda create -n georc python=3.10
   conda activate georc
   ```

3. **Install dependencies from requirements.txt**:
   ```bash
   pip install -r env/requirements.txt
   ```

   The requirements.txt file includes all necessary packages:
   - **PyTorch & Vision**: torch, torchvision, numpy, pillow
   - **Model Libraries**: transformers, bitsandbytes, accelerate
   - **API Clients**: google-genai, azure-ai-vision-imageanalysis, openai, groq
   - **Utilities**: matplotlib, cartopy

## Dataset

The GeoRC dataset is available on Hugging Face. Download it using the following instructions:

### Download Instructions

1. Install the Hugging Face datasets library (if not already installed):
   ```bash
   pip install datasets
   ```

2. Download the dataset from Hugging Face:
   ```bash
   huggingface-cli download mohit-talreja/GeoRC --repo-type dataset --local-dir ./GeoRC_dataset
   ```

3. Alternatively, you can download the dataset programmatically in Python:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("mohit-talreja/GeoRC")
   ```

4. The dataset will be downloaded to `./GeoRC_dataset` (or your specified directory)

For more information, visit: https://huggingface.co/datasets/mohit-talreja/GeoRC

## Generate Script

The `generate.py` script generates reasoning chains for GeoRC challenges using various vision-language models. It processes images and produces detailed geographic reasoning explanations that identify key evidence (lane markings, vegetation, language, architecture, etc.) used to determine location estimates.

### Usage

```bash
python generate.py --type <model_type> --model <model_name> --path <challenge_path> [optional_arguments]
```

### Required Arguments

- `--type`: The type of model to use
  - `hf` - Hugging Face models
  - `gemini` - Google Gemini API
  - `azure` - Azure API

- `--model`: The specific model name to use
  - For Hugging Face: `google`, `llava`, `qwen`, etc.
  - For Gemini: `gemini-1.5-flash`, `gemini-1.5-pro`, etc.
  - For Azure: Your deployed model name

- `--path`: Path to the challenge directory containing images
  - Can be a single challenge directory or parent directory with multiple challenges
  - Images should follow the naming pattern: `{challenge_name}_{1-5}.png`

- `--key`: API key or token
  - For Hugging Face: Your Hugging Face API token
  - For Gemini: Your Google Gemini API key
  - For Azure: Your Azure API key

### Optional Arguments

- `--cache`: Cache directory for model weights (default: `./cache`)
  - Only applicable for Hugging Face models
  - Specifies where downloaded models are stored

- `--endpoint`: API endpoint URL
  - Only required for Azure deployments
  - Example: `https://<resource_name>.openai.azure.com/`

- `--api_version`: API version
  - Only required for Azure deployments
  - Example: `2024-02-15-preview`

### Examples

**Hugging Face Model:**
```bash
python generate.py --type hf --model google --path ./challenges/challenge_1 --key hf_xxxxxxxxxxxx --cache ./models
```

**Gemini API:**
```bash
python generate.py --type gemini --model gemini-1.5-flash --path ./challenges --key sk_xxxxxxxxxxxx
```

**Azure API:**
```bash
python generate.py --type azure --model gpt-4-vision --path ./challenges/challenge_1 --key sk_xxxxxxxxxxxx --endpoint https://myresource.openai.azure.com/ --api_version 2024-02-15-preview
```

### Output

The script generates files named `candidate_reasoning_chain_{model_name}_{image_number}.txt` in each challenge directory. These files contain detailed geographic reasoning explanations for each image in the challenge.

### Using the Dataset with the Generate Script

Once downloaded, use the dataset path as the `--path` argument:

```bash
python generate.py --type hf --model google --path ./GeoRC_dataset --key hf_xxxxxxxxxxxx
```

## Score Script

The `score.py` script evaluates candidate reasoning chains against ground truth chains using various scoring strategies. It supports three different evaluation modes: exact-match/embedding-based key points, LLM-as-a-judge bipartite matching, and VLM-based visual correctness scoring.

### Usage

```bash
python score.py --mode <scoring_mode> --llm_model <model_name> --challenge_path <path> [optional_arguments]
```

### Required Arguments

- `--mode`: Scoring strategy to use
  - `key_points` - Exact-match or embedding-based atomic key point matching
  - `bipartite` - LLM-as-a-judge bipartite matching between candidate and ground truth points
  - `vlm_judge` - Hybrid approach combining VLM visual correctness and LLM reasoning scores

- `--llm_model`: LLM model name for scoring
  - Examples: `meta-llama/Llama-2-7b`, `mistralai/Mistral-7B`, etc.
  - Used for `key_points` and `bipartite` modes

- `--challenge_path`: Path to directory containing challenges to evaluate
  - Can be a single challenge directory or parent directory with multiple challenges
  - Must contain subdirectories with `Human_Chain_3.txt` (ground truth) and candidate files

### Optional Arguments

- `--vlm_model_name`: Vision-Language Model name
  - **Required** if `--mode` is `vlm_judge`
  - Examples: `openai/clip-vit-base-patch32`, `google/vit-base-patch16-224`

- `--pattern`: Filename pattern for candidate reasoning chains (default: `candidate_reasoning_chain_`)
  - The script expects files named `{pattern}{round_number}.txt`
  - Example: if pattern is `model_output_`, it looks for `model_output_1.txt`, `model_output_2.txt`, etc.

- `--suffix`: Suffix for output JSON file (default: `eval_results`)
  - Output file will be named `vlm_scores_{mode}_{suffix}.json`

- `--cache_dir`: Cache directory for model weights (default: `./cache`)
  - Used to store downloaded model weights locally

- `--emb`: Enable embedding-based matching
  - **Only relevant for `key_points` mode**
  - Uses sentence transformers for semantic similarity instead of exact matching
  - Flag to add without argument (e.g., `--emb`)

### Output Format

All modes produce a JSON file with evaluation results containing:
- `challenge`: Challenge name/directory
- `round`: Round number (1-5)
- `precision`: Precision score (0-1)
- `recall`: Recall score (0-1)
- `f1`: F1 score (0-1)

For `vlm_judge` mode, also includes:
- `vlm_correctness`: Visual correctness score from the VLM

### Examples

**Key Points Mode (Exact Match):**
```bash
python score.py --mode key_points --llm_model meta-llama/Llama-2-7b --challenge_path ./GeoRC_dataset
```

**Key Points Mode (With Embeddings):**
```bash
python score.py --mode key_points --llm_model meta-llama/Llama-2-7b --challenge_path ./GeoRC_dataset --emb
```

**Bipartite LLM-as-Judge Mode:**
```bash
python score.py --mode bipartite --llm_model mistralai/Mistral-7B --challenge_path ./challenges --suffix my_eval
```

**VLM Judge Mode (Visual + Reasoning):**
```bash
python score.py --mode vlm_judge --llm_model meta-llama/Llama-2-7b --vlm_model_name google/vit-base-patch16-224 --challenge_path ./GeoRC_dataset --cache_dir ./models
```

### Scoring Modes Explained

#### Key Points Mode
- Extracts atomic facts/points from both candidate and ground truth chains
- **Exact Match** (default): Compares points using exact text matching
- **Embedding-based** (with `--emb`): Uses sentence transformers to compute semantic similarity
- Fast and deterministic; does not require additional LLM calls

#### Bipartite Mode
- Uses an LLM to perform fine-grained matching between candidate and ground truth points
- Evaluates each candidate point against the full ground truth chain
- Evaluates each ground truth point against the full candidate chain
- Computes precision, recall, and F1 based on LLM similarity judgments (0-100 scale)
- More flexible than exact matching; captures paraphrased concepts

#### VLM Judge Mode
- Combines visual and textual evaluation
- Uses a Vision-Language Model to assess visual correctness of the candidate chain
- Integrates VLM correctness score with LLM-based reasoning chain evaluation
- Best for scenarios where visual grounding is important
- **Note**: Requires image paths in the challenge directory

### Ground Truth Format

Challenges must contain a file named `Human_Chain_3.txt` with the ground truth reasoning chain. The script evaluates 5 rounds by default (matches with files `{pattern}1.txt` through `{pattern}5.txt`).

Example directory structure:
```
GeoRC_dataset/
├── challenge_1/
│   ├── challenge_1_1.png
│   ├── challenge_1_2.png
│   ├── challenge_1_3.png
│   ├── challenge_1_4.png
│   ├── challenge_1_5.png
│   ├── Human_Chain_3.txt (ground truth)
│   ├── candidate_reasoning_chain_1.txt
│   ├── candidate_reasoning_chain_2.txt
│   └── ...
└── challenge_2/
    └── ...
```

### Output File

Results are saved as JSON with the naming format: `vlm_scores_{mode}_{suffix}.json`

Example output:
```json
[
    {
        "challenge": "challenge_1",
        "round": 1,
        "precision": 0.85,
        "recall": 0.78,
        "f1": 0.81
    },
    {
        "challenge": "challenge_1",
        "round": 2,
        "precision": 0.92,
        "recall": 0.88,
        "f1": 0.90
    }
]
```