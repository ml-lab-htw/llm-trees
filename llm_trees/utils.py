import os
import re

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from .config import Config
from .io import get_data
from .llms import generate_openrouter_tree, generate_gpt_tree, generate_claude_tree, generate_gemini_tree

# Load the environment variables from .env file
load_dotenv()


def generate_tree(config):
    tree_path = get_tree_path(config)

    if not os.path.exists(tree_path):
        if config.generate_tree_if_missing:
            print(f"$$$ -> {config.method.upper()}: {config.dataset}-{config.iter}")

            if config.method == "openrouter":
                prompting_result = generate_openrouter_tree(config)
            elif config.method in ["gpt-4o", "gpt-o1"]:
                prompting_result = generate_gpt_tree(config)
            elif config.method == "claude":
                prompting_result = generate_claude_tree(config)
            elif config.method == "gemini":
                prompting_result = generate_gemini_tree(config)
            else:
                raise ValueError(f"Unknown method: {config.method}")

            export_prompting_result(tree_path, prompting_result)

            # # avoid quota limit
            # if not ("gpt" in config.method):
            #     time.sleep(10)


def regenerate_tree(config, e=""):
    tree_path = get_tree_path(config)
    print(f"Failed to predict tree {os.path.basename(tree_path)}: {e}")
    if config.regenerating_invalid_trees:
        if os.path.exists(tree_path):
            os.remove(tree_path)
        generate_tree(config)
    else:
        raise e

def predict_tree_from_file(config: Config, X):

    tree_path = get_tree_path(config)

    # load tree from txt file
    with open(tree_path, "r") as file:
        prompting_result = file.read()

    tree_fcn_str = postprocess_prompting_result(config, prompting_result)

    # Safely execute `tree_fcn_str` in a local scope
    local_scope = {}
    try:
        exec(tree_fcn_str, {"np": np, "pd": pd}, local_scope)
        predict = local_scope.get("predict")
        if predict is None:
            print(f"FAILED (no predict function found)")
            return None, None

        if config.force_decision_tree:

            # predict the first sample to get the number of inner nodes
            _, nodes = predict(X.iloc[0].to_dict())
            num_nodes = len(nodes)

            # init predictions and nodes
            y_pred = np.empty(len(X), dtype=int)
            nodes = np.zeros((len(X), num_nodes), dtype=bool)

            for i in range(len(X)):
                y_pred[i], nodes[i, :] = predict(X.iloc[i].to_dict())

            return y_pred, nodes

        else:

            # init predictions and nodes
            y_pred = np.empty(len(X), dtype=int)

            for i in range(len(X)):
                y_pred[i] = predict(X.iloc[i].to_dict())

            return y_pred, None

    except Exception as e:
        raise e

def postprocess_prompting_result(config:Config, prompting_result):
    if config.llm_dialogue:
        second_answer_start_str = 25 * "#"
        if second_answer_start_str in prompting_result:
            prompting_result = prompting_result[prompting_result.find(second_answer_start_str):]

    # split the response text by code blocks
    splitted_result = prompting_result.split("```")

    # if there is only one code block, it is the python code
    if len(splitted_result) < 2:

        # extract the python code from the response text
        start_str ="def predict"
        end_str = "return prediction, nodes"

        if start_str in prompting_result and end_str in prompting_result:
            function_str = prompting_result[prompting_result.find(start_str):prompting_result.find(end_str) + len(end_str)] + "\n"
        else:
            raise ValueError("No predict function found in prompting result")

    else:

        # find the python code block
        python_splits = [split for split in splitted_result if "python\n" in split.lower()]
        function_splits = [split for split in python_splits if "def predict" in split.lower()]
        function_splits = [split for split in function_splits if "return " in split.lower()]

        if len(function_splits) == 0:
            raise ValueError("No predict function found in prompting result")

        function_str = function_splits[0]
        function_str = function_str.replace("python\n", "")

    # extract the python code from the response text
    prompting_result_code = extract_python_code(function_str)

    # if there is "    return prediction, nodes" at the end, cut everything after
    function_end_str = "return prediction, nodes"
    if function_end_str in prompting_result_code:
        prompting_result_code = prompting_result_code[:prompting_result_code.find(function_end_str) + len(function_end_str)]


    # find the first line that does not start with a space
    lines = prompting_result_code.split("\n")
    line_start_offset = 5  # skip the first lines bc the function definition is always longer
    line_idx = None
    for line_idx, line in enumerate(lines[line_start_offset:]):
        if not (line.startswith(" ") or line.startswith("\t")) and not (line == ""):
            break

    # if there is a line that does not start with a space, cut everything
    if line_idx is not None:
        line_idx += line_start_offset
        if line_idx + 1 < len(lines):
            # remove all lines after the first line that does not start with a space
            prompting_result_code = "\n".join(lines[:line_idx])


    tree_fcn_str = fix_predict_fcn_name(prompting_result_code)

    return tree_fcn_str

def fix_predict_fcn_name(tree_str):
    pattern = r"def predict_\w+\(.*\):"
    replacement = "def predict(X):"
    return re.sub(pattern, replacement, tree_str)

def extract_python_code(response_text):
    """
    Extracts Python code from the given ChatGPT response text.

    Parameters:
    response_text (str): The response text from a ChatGPT API call.

    Returns:
    str: Extracted Python code.
    """
    # Define a regular expression pattern to match Python code blocks
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)

    # Find all matches in the response text
    code_blocks = pattern.findall(response_text)

    # if isinstance(code_blocks, list):
    #     code_blocks = code_blocks[0]

    # If no code blocks were found, return the original response text
    if not code_blocks:
        return response_text

    # Join all code blocks into a single string, separating by newlines
    extracted_code = '\n\n'.join(code_blocks)

    return extracted_code.strip()

def get_tree_path(config):

    method = os.environ["OPENROUTER_MODEL"].replace("~", "").replace("/", "_").replace(":", "_")

    if config.force_decision_tree:
        tree_file_name = (f"{config.dataset}_"
                          f"{method}_"
                          f"examples_{config.num_examples}_"
                          f"desc_{config.include_description}_"
                          f"temp_{100 * config.temperature:.0f}_"
                          f"maxdepth_{config.max_tree_depth}_"
                          f"iter_{config.iter}.txt")
    else: # free form
        tree_file_name = (f"{config.dataset}_"
                          f"{method}_"
                          f"free_form_"
                          f"temp_{100 * config.temperature:.0f}_"
                          f"iter_{config.iter}.txt")

    if config.llm_dialogue: # i.e. two step tree generation
        tree_file_name = tree_file_name.replace(".txt", "_dialogue.txt")

    return os.path.join(config.root, "trees", config.dataset, method, tree_file_name)

def export_prompting_result(tree_path, prompting_result):
    if not os.path.exists(os.path.dirname(tree_path)):
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)
    with open(tree_path, "w") as text_file:
        text_file.write(prompting_result)

def count_keys_in_file(keys, file_paths):
    """
    Counts the occurrences of each key in the provided list within the given file.

    Parameters:
    keys (list of str): List of keys to count in the file.
    file_path (str): Path to the Python file.

    Returns:
    dict: Dictionary with keys as the input keys and values as their counts in the file.
    """
    counts = {key: 0 for key in keys}

    try:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                content = file.read()
                for key in keys:
                    counts[key] = content.count(key)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return counts


def get_feature_count(config: Config):

    methods = ["gpt-4o", "gpt-o1", "gemini", "claude"]
    key_counts = {method: {} for method in methods}

    X, y = get_data(config.root, config.dataset)
    data_available = X is not None and y is not None
    if not data_available:
        raise FileNotFoundError(f"Data not found for dataset: {config.dataset}")

    for method in methods:
        config.method = method
        tree_paths = []

        for tree_idx in range(config.num_trees):
            config.iter = tree_idx

            # Generate trees if not available
            generate_tree(config)
            tree_paths.append(get_tree_path(config))

        key_counts[method] = count_keys_in_file(X.keys(), tree_paths)
    return key_counts
