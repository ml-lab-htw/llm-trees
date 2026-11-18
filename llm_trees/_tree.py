"""Two-step LLM response post-processing and executable tree prediction.

Turns the raw two-step dialogue response into a Python ``predict(X)``
function and executes it row-wise. The object-dtype prediction buffer keeps
label dtype, so non-numeric (e.g. string) target labels are supported.
"""

import re

import numpy as np
import pandas as pd

DIALOGUE_SEPARATOR = "#" * 25


def postprocess_prompting_result(prompting_result):
    second_answer_start_str = DIALOGUE_SEPARATOR
    if second_answer_start_str in prompting_result:
        prompting_result = prompting_result[prompting_result.find(second_answer_start_str):]

    # split the response text by code blocks
    splitted_result = prompting_result.split("```")

    # if there is only one code block, it is the python code
    if len(splitted_result) < 2:

        # extract the python code from the response text
        start_str = "def predict"
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
    """Extract the fenced Python code blocks from a raw LLM response.

    Falls back to the response text unchanged when no code fences are found.
    """
    # Define a regular expression pattern to match Python code blocks
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)

    # Find all matches in the response text
    code_blocks = pattern.findall(response_text)

    # If no code blocks were found, return the original response text
    if not code_blocks:
        return response_text

    # Join all code blocks into a single string, separating by newlines
    extracted_code = '\n\n'.join(code_blocks)

    return extracted_code.strip()


def compile_tree_function(tree_fcn_str: str):
    """Exec the generated tree code in a local scope and return its ``predict``."""
    local_scope = {}
    exec(tree_fcn_str, {"np": np, "pd": pd}, local_scope)
    predict = local_scope.get("predict")
    if predict is None:
        raise ValueError("No predict function found in tree code")
    return predict


def _check_node_vector(nodes) -> int:
    """Validate a node vector; return its width.

    Supported representation: a one-dimensional sequence (or array) whose
    entries are ``None`` (off-path), booleans, integers, or floats. Mappings
    and sets are rejected explicitly: iterating them yields keys/elements,
    and NumPy converts them silently when assigned into a matrix row (e.g.
    ``{0: False, 1: True}`` -> ``[1.0, 1.0]``), which would corrupt the
    boolean node matrix. Validation is performed on the ``np.asarray``
    representation -- the same one used when the vector is assigned into a
    NumPy row during embedding construction.
    """
    if nodes is None:
        raise ValueError("Tree function returned nodes=None; nodes must be a sequence")
    if isinstance(nodes, (dict, set, frozenset)):
        raise ValueError(
            f"Tree function returned an unsupported nodes container of type "
            f"{type(nodes).__name__}; nodes must be a one-dimensional sequence"
        )
    try:
        arr = np.asarray(nodes)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Tree function returned unsupported nodes: {nodes!r}") from e
    if arr.ndim != 1:
        raise ValueError(
            f"Tree function returned {arr.ndim}-dimensional nodes; "
            "nodes must be a one-dimensional sequence"
        )
    for v in arr.tolist():
        if v is None or isinstance(v, (bool, np.bool_, int, np.integer, float, np.floating)):
            continue
        raise ValueError(f"Tree function returned unsupported node value: {v!r}")
    return arr.shape[0]


def predict_rows(predict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run the tree function row-wise; returns (y_pred, nodes).

    ``nodes`` is a boolean matrix with off-path (``None``) entries stored as
    ``False``, and ``y_pred`` keeps the label dtype (object) so string
    labels are supported. Every row's node vector is validated (see
    ``_check_node_vector``) before being assigned into the matrix.
    """
    # predict the first sample to get the number of inner nodes
    _, nodes = predict(X.iloc[0].to_dict())
    num_nodes = _check_node_vector(nodes)

    # init predictions and nodes
    y_pred = np.empty(len(X), dtype=object)
    nodes = np.zeros((len(X), num_nodes), dtype=bool)

    for i in range(len(X)):
        y_pred[i], row = predict(X.iloc[i].to_dict())
        if _check_node_vector(row) != num_nodes:
            raise ValueError(f"Inconsistent node vector widths across rows: {num_nodes} vs {len(row)}")
        nodes[i, :] = row

    return y_pred, nodes
