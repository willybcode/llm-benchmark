import argparse
from typing import List

from ollama import Client
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime

#  string manip
import re

DEFAULT_HOST = 'http://localhost:11434'
class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided, potentially due to prompt caching. For more info, see https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(model_name: str, prompt: str, verbose: bool, client: Client) -> OllamaResponse:

    last_element = None

    if verbose:
        stream = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    # with open("data/ollama/ollama_res.json", "w") as outfile:
    #     outfile.write(json.dumps(last_element, indent=4))

    return last_element


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)

def get_full_model_list(client) -> List[str]:
    """
    Get a list of all available Ollama model names.
    Returns:
        List[str]: The list of model names.
    """
    models = client.list().get("models", [])
    # print(models)
    return [model["model"] for model in models]

def show_models_list(model_names: List[str]) -> None:
    """
    Print a numbered list of model names.
    Args:
        model_names (List[str]): The list of model names to print.
    Returns:
        None
    """
    print("\nAvailable models:\n")
    for i, name in enumerate(model_names):
        print(f"{i+1}. {name}")
        
def get_user_choice(choices: List[str], choice_text: str) -> str:
    """Get user choice for prompts"""
    while True:
        prompt_choice = input(choice_text).strip().upper()
        if prompt_choice in choices:
            return prompt_choice
        print("\nInvalid choice. Please try again.")

def validate_input(user_input):
    """Remove spaces and validate input format"""
    user_input = user_input.replace(" ", "")
    if not re.match("^[0-9,]+$", user_input):
        raise ValueError("Invalid input. Please enter only digits and commas.")
    return user_input

def get_benchmark_models(models_list: List[str], models_to_use: List[str] = [], models_to_skip: List[str] = []) -> List[str]:
    """
    Get a list of benchmark model names.
    
    Args:
        models_list (List[str]): A list of all available models
        models_to_use (List[str]): A list of model names to use exclusively. If provided,
            no other models will be included.
        models_to_skip (List[str]): A list of model names to exclude from the results.
    
    Returns:
        List[str]: The list of selected model names.
    
    Raises:
        ValueError: If both `models_to_use` and `models_to_skip` are provided.
    """

    model_names = models_list.copy()

    if len(models_to_use) > 0 and len(models_to_skip) > 0:
        raise ValueError("Cannot provide both 'Models to use' and 'Models to skip' at the same time")
    if len(models_to_use) > 0:
        model_names = [model for model in model_names if model in models_to_use]
    elif len(models_to_skip) > 0:
        model_names = [model for model in model_names if model not in models_to_skip]
    # print(f"Evaluating models: {model_names}\n")
    return model_names

def get_custom_prompts():
    """Get custom prompts from user"""
    PROMPT_SEPARATOR = '|'

    print(f"\nCustom prompts should be separated by {PROMPT_SEPARATOR}. Quotes are optional. e.g.: prompt1 {PROMPT_SEPARATOR} \"prompt 2\" {PROMPT_SEPARATOR} prompt3")
    keep_prompting = True
    while keep_prompting:
        user_input = input(f"Enter custom prompts ({PROMPT_SEPARATOR}-separated):\n\n>> ")
        
        # Split input by separator
        prompts = user_input.split(PROMPT_SEPARATOR)
        
        # Strip whitespace and remove surrounding quotes for each prompt
        prompts = [prompt.strip().strip('"') for prompt in prompts]
        
        # Remove empty prompts
        prompts = [prompt for prompt in prompts if prompt]
        
        if prompts:
            return prompts
        else:
            print("\nError: No valid prompts entered. Please try again.")

def validate_selection_range(user_input : str,max : int):
    """Validate selection range"""
    
    for i in user_input.split(','):
        if not 1 <= (i := int(i)) <= max:
            print(f"\nError: Invalid selection '{i}' Please try again.")
            return False
    return user_input

def validate_host(user_host: str):
    """Validate host"""
    pattern = r"^(http|https)://((([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})|(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))(:\d{1,5})?$"
    if not re.match(pattern, user_host):
        print(f"\nError: Invalid host '{user_host}'. Please try again. Examples of valid hosts are http://localhost, http://localhost:11434, http://10.0.0.10:1156, https://myserver.dev")
        return False
    return user_host
def main():
    default_prompts =[
        "Why is the sky blue?",
        "Write a report on the financials of Microsoft",
    ]
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models.",
        epilog="See https://github.com/willybcode/llm-benchmark for more information.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Use all available models",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-u",
        "--use-models",
        nargs="*",
        default=[],
        help="List of model names to use exclusively. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=default_prompts,
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=DEFAULT_HOST,
        help="Ollama API host example: http://localhost:11434",
    )
    
    args = parser.parse_args()
    verbose = args.verbose
    all_models: bool = args.all
    models_to_skip = args.skip_models
    models_to_use = args.use_models
    prompts = args.prompts
    remote= args.remote
    if not validate_host(remote):
        return
    client = Client(host=remote)

    models_list = get_full_model_list(client)
    selected_models = []
    if not models_list:
        print(f"\nNo models found with ollama. Pull some models first")
        return
    

    if not all_models and len(models_to_skip) == 0 and len(models_to_use) == 0:
        # interactive = True
        print("\nWhat would you like to do?")
        user_choice = get_user_choice(['A','B','C'],"A) Select models to benchmark\nB) Select models to skip in benchmark\nC) Run benchmark on all models\n\n>> ")

        if user_choice == 'A':
            show_models_list(models_list)
            user_input = input("\nEnter a comma separated list of model numbers to use (e.g., 1,2,3):\n\n>> ")
            
            try:
                user_input = validate_input(user_input)
                if not (validate_selection_range(user_input,len(models_list))):
                    return
                models_to_use = [models_list[int(i)-1] for i in user_input.split(',')]
            except ValueError:
                print("\nError: Invalid input. Please try again.")
                return
        elif user_choice == 'B':
            show_models_list(models_list)
            user_input = input("\nEnter a comma separated list of model numbers to skip (e.g., 1,2,3):\n\n>> ")
            
            try:
                user_input = validate_input(user_input)
                if not (validate_selection_range(user_input,len(models_list))):
                    return
                models_to_skip = [models_list[int(i)-1] for i in user_input.split(',')]
            except ValueError:
                print("\nError: Invalid input. Please try again.")
                return
        elif user_choice == 'C':
            all_models = True
        
        if not verbose:
            verbose_choice = input("\nVerbose? [y/n] : ")
            if verbose_choice.strip().lower() == 'y':
                verbose = True
            
        if not prompts:
            prompts=default_prompts
            
        prompt_choice = get_user_choice(['A','B'],"\nA) Use %s prompts\nB) Use Custom prompts\n\n>> " % ('Default' if prompts == default_prompts else 'Currently set')).strip().upper()
            
            
        if prompt_choice == 'A':
            if not prompts:
                prompts = default_prompts
        elif prompt_choice == 'B':
            prompts = get_custom_prompts()
        
    # selected_models = models_list if all_models else get_benchmark_models(models_list, models_to_use, models_to_skip)
    if all_models:
        models_to_use = selected_models = models_list
    else:
        selected_models = get_benchmark_models(models_list, models_to_use, models_to_skip)
        
    if not selected_models:
        print('No models selected.')
        return

    print(
        f"\nVerbose: {verbose}\nUse models: {models_to_use}\nSkip models: {models_to_skip}\nPrompts: {prompts}"
    )
    if selected_models == models_list:
        print("\nRunning benchmark on all available models")
    
            
    benchmarks = {}
    for model_name in selected_models:
        responses: List[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
                print("Response:")
            response = run_benchmark(model_name, prompt, verbose=verbose, client=client)
            responses.append(response)

            if verbose:
                print(response.message.content)
                inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M llama2:latest --prompts "What color is the sky" "Write a report on the financials of Microsoft"
