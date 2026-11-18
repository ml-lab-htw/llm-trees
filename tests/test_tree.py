"""Tests for llm_trees._tree: response post-processing and exec.

The post-processing happy path (separator split, fenced-code extraction) is
exercised end-to-end by the retry-loop tests in tests/test_utils.py, and
``predict_rows`` by the embedding tests with synthetic scripted trees.
"""

import pytest

from llm_trees._tree import compile_tree_function, postprocess_prompting_result


def test_postprocess_no_predict_function_raises():
    with pytest.raises(ValueError, match="No predict function found"):
        postprocess_prompting_result("sorry, I cannot help with that")


def test_compile_tree_function_missing_predict_raises():
    with pytest.raises(ValueError):
        compile_tree_function("def something_else():\n    return 1\n")
