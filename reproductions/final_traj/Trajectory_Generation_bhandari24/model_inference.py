import torch
import os
import sys
import json
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None
from utils import (
    read_dialogs_from_file,
    format_tokens,
    create_update_prompt,
)
from openai import (
    OpenAI,
    AzureOpenAI
)
try:
    import replicate
except ModuleNotFoundError:
    replicate = None


def require_optional_dependency(module, package_name):
    if module is None:
        raise ModuleNotFoundError(
            f"Optional dependency '{package_name}' is required for this model backend. "
            f"Install it or choose another --use_model value."
        )


def read_secret(env_name, file_name):
    value = os.environ.get(env_name)
    if value:
        return value.strip()
    if os.path.exists(file_name):
        return open(file_name).read().strip()
    raise FileNotFoundError(
        f"Set {env_name} or create {file_name} before using this backend."
    )


def conduct_survey(
    model,
    tokenizer,
    max_new_tokens=256,
    prompt_file: str = None,
    do_sample: bool = True,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
    survey_questions: json = None,
    **kwargs
):
    if prompt_file is not None:
        if type(prompt_file) is list:
            dialogs = prompt_file
        else:
            assert os.path.exists(
                prompt_file
            ), f"Provided Prompt file does not exist {prompt_file}"

            dialogs = read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    # Prepare answers
    answers = survey_questions.copy()
    answers.insert(0, prompt_file[0][0])

    with torch.no_grad():
        ans_output = None
        for idx, question in enumerate(survey_questions):
            if idx == 0:
                dialogs = create_update_prompt(dialogs, question, ans_output)
            else:
                dialogs = create_update_prompt(dialogs, question, ans_output)
            chat = format_tokens(dialogs, tokenizer)
            tokens = torch.tensor(chat[0]).long()
            tokens = tokens.unsqueeze(0)
            tokens = tokens.to("cuda:0")
            outputs = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            full_output = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            ans_output = tokenizer.decode(
                outputs[0][len(chat[0]):],
                skip_special_tokens=True
            )

            answers[idx+1]['full_output'] = full_output
            answers[idx+1]['ans_output'] = ans_output

            answers[idx+1]['full_output'] = full_output
            answers[idx+1]['ans_output'] = ans_output
    return answers


def conduct_completion(
    model,
    tokenizer,
    max_new_tokens=512,
    prompt_file: str = None,
    do_sample: bool = True,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
):

    answers = {
        'input': prompt_file,
    }
    with torch.no_grad():
        tokens = tokenizer.encode(prompt_file, return_tensors="pt")
        tokens = tokens.to("cuda:0")
        outputs = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )

        answers['full_output'] = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        answers['ans_output'] = tokenizer.decode(
            outputs[0][len(prompt_file):],
            skip_special_tokens=True
        )

    return answers


def _first_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def conduct_qwen_local_completion(
    model,
    tokenizer,
    max_new_tokens=1024,
    prompt_file: str = None,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
):
    answers = {
        'input': prompt_file,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Complete the travel diary. Return only the final markdown table "
                "with columns Place Visited, Arrival Time, Departure Time, and Location Type. "
                "Do not include reasoning or explanations."
            ),
        },
        {"role": "user", "content": prompt_file},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = prompt_file

    inputs = tokenizer(prompt, return_tensors="pt")
    device = _first_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    answers['ans_output'] = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    answers['full_output'] = prompt + answers['ans_output']
    return answers


def conduct_palm_completion(
    model,
    prompt_file,
    max_new_tokens=512,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
):
    require_optional_dependency(genai, "google-generativeai")

    answers = {
        'input': prompt_file,
    }
    completion = genai.generate_text(
        model=model,
        prompt=prompt_file,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_new_tokens,
    )

    answers['ans_output'] = completion.result
    return answers


def conduct_gpt3_completion(
    prompt_file,
):
    openai_key = read_secret("OPENAI_API_KEY", "openai_key_new")
    client = OpenAI(api_key=openai_key)
    answers = {
        'input': prompt_file,
    }
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": f"{prompt_file}"}
      ]
    )
    answers['ans_output'] = completion.choices[0].message.content
    return answers


def conduct_gpt4_completion(
    prompt_file,
):

    openai_key = read_secret("AZURE_OPENAI_API_KEY", "openai_key_azure")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://olympia.openai.azure.com/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "mtob-gpt4-turbo")
    client = AzureOpenAI(
        api_key=openai_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint,
    )
    answers = {
        'input': prompt_file,
    }
    completion = client.chat.completions.create(
      model=deployment,
      messages=[
        {"role": "user", "content": f"{prompt_file}"}
      ]
    )
    answers['ans_output'] = completion.choices[0].message.content
    return answers


def conduct_palm_agent(
    model,
    prompt,
    max_new_tokens=1024,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
):
    survey = genai.generate_text(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_output_tokens=max_new_tokens,
    )

    return survey.result


def conduct_agent(
    model,
    tokenizer,
    max_new_tokens=512,
    prompt_file: str = None,
    do_sample: bool = True,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
):
    with torch.no_grad():
        tokens = tokenizer.encode(prompt_file, return_tensors="pt")
        tokens = tokens.to("cuda:0")
        outputs = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def conduct_llama2_70b_api_completion(
    model,
    prompt_file,
    max_new_tokens=512,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
):
    require_optional_dependency(replicate, "replicate")

    api = replicate.Client(
        api_token=read_secret("REPLICATE_API_TOKEN", "replicate_api_key"),
        timeout=None
    )
    answers = {
        'input': prompt_file,
    }
    output = api.run(
        model,
        input={
            "top_k": top_k,
            "top_p": top_p,
            "prompt": prompt_file,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "prompt_template": "{prompt}",
        },
    )
    answers['ans_output'] = output
    return answers


def conduct_completion_llama2_70b(
    llm,
    sampling_params,
    prompt_file,
):

    answers = {
        'input': prompt_file,
    }
    response = llm.generate(prompt_file, sampling_params)
    answers['ans_output'] = response[0].outputs[0].text

    return answers


def conduct_gemini_completion(
    model,
    prompt,
    max_new_tokens=512,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
):
    require_optional_dependency(genai, "google-generativeai")
    answers = {
        'input': prompt,
    }
    completion = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
        )
    )
    if completion.candidates[0].finish_reason.value != 1:
        completion = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
            )
        )

    answers['ans_output'] = completion.text
    return answers
