# Root conftest: ensures the package root is on sys.path for pytest (all import modes).

import pytest

from llm_trees._tree import compile_tree_function
from llm_trees._tree_ast import parse_tree_function

DIALOGUE_SEPARATOR = "#" * 25


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in an isolated CWD.

    ``fit`` writes the step prompts by default (``prompt_file=True``), so an
    isolated CWD keeps the prompt-file side effect out of the repository.
    """
    monkeypatch.chdir(tmp_path)


def make_scripted_tree(fcn_source):
    """Compile a tree function source into the ``(code, fn, pt)`` tuple form
    used by the embedding transformer (deterministic; no LLM involved)."""
    fn = compile_tree_function(fcn_source)
    return (fcn_source, fn, parse_tree_function(fcn_source))


class ScriptedClient:
    """LLM client stub returning a scripted raw two-step response each call."""

    def __init__(self, raw_response):
        self.raw_response = raw_response
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        return self.raw_response


def make_float_tree_client(cols):
    """Build a ScriptedClient whose step-2 function returns float predictions.

    The generated tree is of the form
    ``return float(X[c0] * 2 + X[c1]), [X[c0] <= 1.0]`` — a valid executable
    ``predict`` whose output is numeric (regression) and whose structure is a
    plain return (not the nodes/if form), so ``tree_`` will be ``None``.
    """
    c0, c1 = cols[0], cols[1]
    code = (
        "```python\n"
        "def predict(X: dict):\n"
        f'    return float(X["{c0}"] * 2 + X["{c1}"]), [X["{c0}"] <= 1.0]\n'
        "```"
    )
    raw = (
        "A decision tree for the target.\n"
        "\n\n\n"
        + DIALOGUE_SEPARATOR
        + "\n\n\n"
        + code
    )
    return ScriptedClient(raw)
