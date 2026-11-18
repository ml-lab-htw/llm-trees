"""Tests for llm_trees._prompts: parameterized two-step prompt construction."""

import pytest

from tests.fixtures import PROMPT_TEMPLATE
from llm_trees._prompts import (
    build_first_prompt,
    build_second_prompt,
    complexity_clause,
    get_role_prompt,
)
from llm_trees._prompt_examples import get_example

FEATURE_NAMES = ["signal", "amplitude", "frequency"]
FEATURES = {
    "signal": "A synthetic signal measurement.",
    "amplitude": "The amplitude of the synthetic signal.",
    "frequency": "The frequency of the synthetic signal.",
}
TARGET = {"outcome": "synthetic outcome (0 = class A / 1 = class B / 2 = class C)"}


def _base_first_prompt(**overrides):
    kwargs = dict(
        target=TARGET,
        features={name: "" for name in FEATURE_NAMES},
        max_tree_depth=2,
        num_features=None,
        num_examples=1,
        use_role_prompt=False,
    )
    kwargs.update(overrides)
    return build_first_prompt(**kwargs)


def test_first_prompt_matches_spec_template():
    # base case must equal the spec template rendered with target/features/depth=2
    expected = (
        PROMPT_TEMPLATE.replace("{max_tree_depth}", "2")
        .replace("{target_description}", "synthetic outcome (0 = class A / 1 = class B / 2 = class C)")
        .replace("{features}", ", ".join(FEATURE_NAMES))
    )
    assert _base_first_prompt() == expected


def test_complexity_clauses_four_combinations():
    assert complexity_clause(2, None) == " with a maximum depth of 2"
    assert complexity_clause(None, 3) == " with at most 3 distinct features"
    assert complexity_clause(2, 3) == " with a maximum depth of 2 and at most 3 distinct features"
    assert complexity_clause(None, None) == ""


@pytest.mark.parametrize(
    "max_tree_depth, num_features, expected_clause",
    [
        (2, None, "with a maximum depth of 2 using the most important features"),
        (None, 3, "with at most 3 distinct features using the most important features"),
        (3, 3, "with a maximum depth of 3 and at most 3 distinct features using the most important features"),
        (None, None, "using the most important features"),
    ],
)
def test_first_prompt_complexity_clauses(max_tree_depth, num_features, expected_clause):
    prompt = _base_first_prompt(max_tree_depth=max_tree_depth, num_features=num_features)
    assert f"build a decision tree {expected_clause}" in prompt


def test_first_prompt_renders_target_and_feature_descriptions():
    prompt = _base_first_prompt(features=FEATURES)
    assert "The tree should predict the synthetic outcome (0 = class A / 1 = class B / 2 = class C)." in prompt
    assert (
        "Features: signal: A synthetic signal measurement., "
        "amplitude: The amplitude of the synthetic signal., "
        "frequency: The frequency of the synthetic signal."
    ) in prompt


def test_first_prompt_without_descriptions_is_names_only():
    base = _base_first_prompt()
    assert "Features: signal, amplitude, frequency" in base
    assert _base_first_prompt(features={name: "" for name in FEATURE_NAMES}) == base


def test_first_prompt_num_examples_phrase_swap():
    assert "I first give an example below." in _base_first_prompt()
    assert "I first give 3 examples below." in _base_first_prompt(num_examples=3)
    assert "I first give no examples below." in _base_first_prompt(num_examples=0)


def test_first_prompt_role_prompt_prefix():
    prompt = _base_first_prompt(use_role_prompt=True)
    assert prompt.startswith(get_role_prompt())
    assert "You are a domain expert" in prompt


def test_second_prompt_contains_keys_target_descriptions_and_example():
    prompt = build_second_prompt(target=TARGET, features=FEATURES, num_examples=1, max_tree_depth=2)
    assert "Now, format this decision tree as a Python function that returns a single prediction" in prompt
    assert "If your decision tree has placeholders for thresholds, try to fill them with meaningful values." in prompt
    # feature keys are rendered as the list repr
    assert str(FEATURE_NAMES) in prompt
    assert "outcome: synthetic outcome (0 = class A / 1 = class B / 2 = class C)" in prompt
    assert "signal: A synthetic signal measurement." in prompt
    assert "Here is an example of what the function should look like:" in prompt
    # depth-2 example content (get_example(0, 2) == example_0_depth_2)
    assert get_example(0, 2) in prompt


def test_second_prompt_num_examples_multiple():
    prompt = build_second_prompt(target=TARGET, features=FEATURES, num_examples=3, max_tree_depth=2)
    assert "Here are 3 examples of what the function should look like:" in prompt
    for i in range(3):
        assert get_example(i, 2) in prompt


def test_second_prompt_descriptions_block():
    prompt = build_second_prompt(target=TARGET, features=FEATURES, num_examples=1, max_tree_depth=None)
    assert "Here is a description of the features with optional feature encoding or ranges:" in prompt
    # None depth -> depth-less example (example_0_depth_5 without the "of depth 5" label)
    assert get_example(0, None) in prompt
