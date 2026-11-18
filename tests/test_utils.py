"""Tests for llm_trees._utils: input validation, label mapping, retry machinery."""

import numpy as np
import pandas as pd
import pytest

from llm_trees._tree import DIALOGUE_SEPARATOR
from llm_trees._tree_ast import parse_tree_function
from llm_trees._utils import (
    _check_node_vector,
    check_complexity_constraints,
    check_complexity_param,
    check_feature_names,
    generate_tree_with_retry,
    is_numeric_prediction,
    map_label,
    resolve_feature_names,
)


class FakeClient:
    """Returns scripted responses; repeats the last one once exhausted."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        return self.responses[min(self.calls, len(self.responses)) - 1]


DEPTH3_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    nodes = 3 * [None]\n'
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    '    nodes[2] = X["c"] <= 3.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        if nodes[1]:\n"
    "            if nodes[2]:\n"
    "                prediction = 1\n"
    "            else:\n"
    "                prediction = 2\n"
    "        else:\n"
    "            prediction = 3\n"
    "    return prediction, nodes\n"
    "```"
)

DEPTH2_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        if nodes[1]:\n"
    "            prediction = 1\n"
    "        else:\n"
    "            prediction = 2\n"
    "    return prediction, nodes\n"
    "```"
)

MISSING_KEY_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["nonexistent"] <= 1.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        prediction = 1\n"
    "    return prediction, nodes\n"
    "```"
)

INVALID_LABEL_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    prediction = "unknown"\n'
    "    return prediction, [True]\n"
    "```"
)

# Unparsable (if-chain tests raw conditions instead of ``nodes[i]``) while
# referencing two features — the num_features limit must still be enforced.
UNPARSABLE_TWO_FEATURE_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    prediction = 0 if X["a"] + X["b"] < 3.0 else 1\n'
    '    return prediction, [X["a"] <= 1.0]\n'
    "```"
)

# Unparsable, genuinely deep (3 nested conditions) tree — the max_tree_depth
# limit must not be bypassable just because the structure is unrecognized.
UNPARSABLE_DEEP_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 3 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    '    nodes[2] = X["c"] <= 3.0\n'
    '    if X["a"] <= 1.0:\n'
    "        prediction = 0\n"
    "    else:\n"
    '        if X["b"] <= 2.0:\n'
    '            if X["c"] <= 3.0:\n'
    "                prediction = 1\n"
    "            else:\n"
    "                prediction = 2\n"
    "        else:\n"
    "            prediction = 3\n"
    "    return prediction, nodes\n"
    "```"
)


def test_resolve_names_dataframe_keys_must_match_columns():
    X = pd.DataFrame({"a": [1], "b": [2]})
    assert list(resolve_feature_names(X, {"a": "d", "b": "d"})) == ["a", "b"]
    with pytest.raises(ValueError, match="unseen: \\['c'\\]"):
        resolve_feature_names(X, {"a": "d", "b": "d", "c": "d"})
    with pytest.raises(ValueError, match="missing: \\['b'\\]"):
        resolve_feature_names(X, {"a": "d", "c": "d"})


def test_resolve_names_array_keys_define_names():
    names = resolve_feature_names(np.ones((3, 2)), {"x": "d", "y": "d"})
    assert list(names) == ["x", "y"]
    assert names.dtype == object
    with pytest.raises(ValueError, match="features has 1 entries"):
        resolve_feature_names(np.ones((3, 2)), {"x": "d"})


def test_resolve_names_rejects_duplicate_columns():
    # duplicate column names would silently discard values in the row-wise
    # X.iloc[i].to_dict() conversion
    X = pd.DataFrame(np.ones((3, 2)), columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_feature_names(X, {"a": "d"})


def test_check_feature_names():
    names = np.array(["a", "b"], dtype=object)
    check_feature_names(pd.DataFrame({"a": [1], "b": [2]}), names)
    check_feature_names(np.ones((4, 2)), names)
    with pytest.raises(ValueError, match="missing"):
        check_feature_names(pd.DataFrame({"a": [1]}), names)
    with pytest.raises(ValueError, match="unseen"):
        check_feature_names(pd.DataFrame({"a": [1], "c": [2]}), names)
    with pytest.raises(ValueError, match="features"):
        check_feature_names(np.ones((4, 3)), names)


def test_map_label_direct_and_positional_and_invalid():
    assert map_label("normal", np.array(["abnormal", "normal"])) == "normal"
    assert map_label(0, np.array(["alpha", "beta", "gamma"])) == "alpha"
    assert map_label(2, np.array(["a", "b", "c"])) == "c"
    with pytest.raises(ValueError, match="Invalid prediction label"):
        map_label("delta", np.array(["normal", "abnormal"]))
    with pytest.raises(ValueError, match="Invalid prediction label"):
        map_label(5, np.array(["a", "b"]))


def test_is_numeric_prediction():
    assert is_numeric_prediction(0)
    assert is_numeric_prediction(3.14)
    assert is_numeric_prediction(np.int64(4))
    assert is_numeric_prediction(np.float32(0.5))
    assert not is_numeric_prediction("3")
    assert not is_numeric_prediction(float("nan"))
    assert not is_numeric_prediction(float("inf"))
    assert not is_numeric_prediction(True)


def test_check_complexity_constraints_depth_and_features():
    pt3 = parse_tree_function(DEPTH3_CODE.split("```python", 1)[1].split("```", 1)[0])
    pt2 = parse_tree_function(DEPTH2_CODE.split("```python", 1)[1].split("```", 1)[0])
    assert pt3 is not None and pt3.depth == 3 and len(pt3.features) == 3
    assert pt2 is not None and pt2.depth == 2
    with pytest.raises(ValueError, match="max_tree_depth=2"):
        check_complexity_constraints(pt3, 2, None)
    with pytest.raises(ValueError, match="num_features=2"):
        check_complexity_constraints(pt3, None, 2)
    check_complexity_constraints(pt3, 3, 3)  # at the limit is allowed
    check_complexity_constraints(pt2, 2, 2)
    check_complexity_constraints(None, None, None)  # unparseable, no limits -> accepted


def test_unparsable_tree_rejected_when_num_features_configured():
    code = UNPARSABLE_TWO_FEATURE_CODE.split("```python", 1)[1].split("```", 1)[0]
    pt = parse_tree_function(code)
    assert pt is None  # unrecognized form, yet it references two features
    with pytest.raises(ValueError, match="could not be parsed"):
        check_complexity_constraints(pt, None, 1)
    # end-to-end: the retry loop must not accept it when num_features=1
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 2.0]})
    client = FakeClient([UNPARSABLE_TWO_FEATURE_CODE])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, num_features=1, num_retry_llm=2))
    assert client.calls == 2


def test_unparsable_tree_rejected_when_max_tree_depth_configured():
    code = UNPARSABLE_DEEP_CODE.split("```python", 1)[1].split("```", 1)[0]
    pt = parse_tree_function(code)
    assert pt is None
    with pytest.raises(ValueError, match="could not be parsed"):
        check_complexity_constraints(pt, 1, None)
    # end-to-end: a deep unrecognized structure cannot bypass max_tree_depth
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]})
    client = FakeClient([UNPARSABLE_DEEP_CODE])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(
            client=client, **_retry_kwargs(X=X, max_tree_depth=2, classes=np.array([0, 1, 2, 3]), num_retry_llm=2)
        )
    assert client.calls == 2


def test_unparsable_tree_accepted_without_limits():
    code = UNPARSABLE_TWO_FEATURE_CODE.split("```python", 1)[1].split("```", 1)[0]
    pt = parse_tree_function(code)
    assert pt is None
    check_complexity_constraints(pt, None, None)  # nothing configured -> nothing to enforce


NEGATIVE_LEAF_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    "        prediction = -1.0\n"
    "    else:\n"
    "        prediction = 2.0\n"
    "    return prediction, nodes\n"
)

NEGATIVE_LEAF_DEEP_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        if nodes[1]:\n"
    "            prediction = -1.0\n"
    "        else:\n"
    "            prediction = 0.0\n"
    "    else:\n"
    "        prediction = 1.0\n"
    "    return prediction, nodes\n"
)


def test_negative_leaf_tree_passes_configured_constraints():
    pt = parse_tree_function(NEGATIVE_LEAF_CODE)
    assert pt is not None and pt.depth == 1 and pt.features == ("a",)
    check_complexity_constraints(pt, 1, None)  # depth-only
    check_complexity_constraints(pt, None, 1)  # feature-only
    check_complexity_constraints(pt, 1, 1)  # both
    pt_deep = parse_tree_function(NEGATIVE_LEAF_DEEP_CODE)
    assert pt_deep is not None and pt_deep.depth == 2
    with pytest.raises(ValueError, match="max_tree_depth=1"):
        check_complexity_constraints(pt_deep, 1, None)


def test_negative_leaf_regression_tree_accepted_end_to_end():
    client = FakeClient(["```python\n" + NEGATIVE_LEAF_CODE + "```"])
    X = pd.DataFrame({"a": [0.0, 2.0]})
    code, pt, fn = generate_tree_with_retry(
        client=client, **_retry_kwargs(X=X, max_tree_depth=1, classes=None, numeric=True)
    )
    assert pt is not None and pt.depth == 1
    preds = [fn(X.iloc[i].to_dict())[0] for i in range(len(X))]
    assert preds == [-1.0, 2.0]


X_GET_FEATURE_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X.get("a") <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        if nodes[1]:\n"
    "            prediction = 1\n"
    "        else:\n"
    "            prediction = 2\n"
    "    return prediction, nodes\n"
)


def test_x_get_feature_tree_rejected_when_num_features_exceeded():
    pt = parse_tree_function(X_GET_FEATURE_CODE)
    assert pt is not None and pt.features == ("a", "b")  # both access forms counted
    with pytest.raises(ValueError, match="num_features=1"):
        check_complexity_constraints(pt, None, 1)
    # end-to-end: two features (one via X.get) cannot pass num_features=1
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 2.0]})
    client = FakeClient(["```python\n" + X_GET_FEATURE_CODE + "```"])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, num_features=1, num_retry_llm=2))
    assert client.calls == 2


NODES_NONE_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    prediction = 0 if X["a"] < 1.0 else 1\n'
    "    return prediction, None\n"
    "```"
)

INCONSISTENT_WIDTH_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    if X["a"] <= 1.0:\n'
    "        return 0, [True]\n"
    "    else:\n"
    "        return 1, [True, False]\n"
    "```"
)

DICT_NODES_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = {0: True, 1: False}\n"
    "    prediction = 0\n"
    "    return prediction, nodes\n"
    "```"
)


def test_nodes_none_rejected_in_retry_loop():
    X = pd.DataFrame({"a": [0.0, 2.0]})
    client = FakeClient([NODES_NONE_CODE])
    with pytest.raises(ValueError, match="nodes"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, classes=None, numeric=True, num_retry_llm=2))
    assert client.calls == 2


def test_inconsistent_node_width_rejected_in_retry_loop():
    X = pd.DataFrame({"a": [0.0, 2.0]})  # one row per branch
    client = FakeClient([INCONSISTENT_WIDTH_CODE])
    with pytest.raises(ValueError, match="width"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, classes=None, numeric=True, num_retry_llm=2))
    assert client.calls == 2


def test_check_node_vector_rejects_unsupported_containers():
    # supported one-dimensional representations, incl. a constant tree's
    # empty node vector
    assert _check_node_vector([True, False]) == 2
    assert _check_node_vector((True,)) == 1
    assert _check_node_vector(np.array([True, True, False])) == 3
    assert _check_node_vector([]) == 0
    # mappings and sets iterate over keys/elements and NumPy converts them
    # silently (e.g. {0: False, 1: True} -> [1.0, 1.0]), corrupting the
    # boolean node matrix
    with pytest.raises(ValueError, match="dict"):
        _check_node_vector({0: True, 1: False})
    with pytest.raises(ValueError, match="set"):
        _check_node_vector({True})
    with pytest.raises(ValueError, match="set"):
        _check_node_vector(frozenset({True}))
    # multi-dimensional and scalar representations are not node vectors
    with pytest.raises(ValueError, match="one-dimensional"):
        _check_node_vector([[True], [False]])
    with pytest.raises(ValueError, match="one-dimensional"):
        _check_node_vector(True)


def test_dict_nodes_rejected_in_retry_loop():
    # a dict node vector must be rejected during validation and retried,
    # not accepted and silently converted by NumPy downstream
    X = pd.DataFrame({"a": [0.0, 2.0]})
    client = FakeClient([DICT_NODES_CODE])
    with pytest.raises(ValueError, match="nodes"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, classes=None, numeric=True, num_retry_llm=2))
    assert client.calls == 2


def test_malformed_nodes_then_valid_response_recovers():
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 2.0]})
    client = FakeClient([INCONSISTENT_WIDTH_CODE, DEPTH2_CODE])
    code, pt, fn = generate_tree_with_retry(
        client=client, **_retry_kwargs(X=X, classes=None, numeric=True, num_retry_llm=3)
    )
    assert client.calls == 2
    assert pt is not None and pt.depth == 2
    assert [fn(X.iloc[i].to_dict())[0] for i in range(len(X))] == [0, 1]


def _retry_kwargs(**overrides):
    kwargs = dict(
        first_prompt="f",
        second_prompt="s",
        X=pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 2.0], "c": [0.0, 2.0]}),
        classes=np.array([0, 1, 2]),
        max_tree_depth=None,
        num_features=None,
        num_retry_llm=3,
        numeric=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_retry_succeeds_after_failures():
    # the valid response carries the full two-step shape (separator between
    # the step-1 text and the fenced step-2 code)
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 2.0]})
    valid = "a step one tree\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + DEPTH2_CODE
    client = FakeClient(["garbage 1", "garbage 2", valid])
    code, pt, fn = generate_tree_with_retry(
        client=client,
        first_prompt="f",
        second_prompt="s",
        X=X,
        classes=np.array([0, 1, 2]),
        max_tree_depth=None,
        num_features=None,
        num_retry_llm=3,
        numeric=False,
    )
    assert client.calls == 3
    assert pt is not None and pt.depth == 2
    y, n = fn(X.iloc[0].to_dict())
    assert y in (0, 1, 2)


def test_retry_exhaustion_raises_with_count():
    client = FakeClient(["garbage 1", "garbage 2"])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(client=client, **_retry_kwargs(num_retry_llm=2))
    assert client.calls == 2


def test_retry_on_invalid_label():
    client = FakeClient([INVALID_LABEL_CODE])
    X = pd.DataFrame({"a": [0.0, 2.0]})
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(
            client=client,
            **_retry_kwargs(X=X, num_retry_llm=2),
        )
    assert client.calls == 2


def test_retry_on_constraint_violation():
    client = FakeClient([DEPTH3_CODE, DEPTH2_CODE])
    code, pt, fn = generate_tree_with_retry(
        client=client, **_retry_kwargs(max_tree_depth=2, num_retry_llm=3)
    )
    assert client.calls == 2
    assert pt.depth == 2


def test_predict_missing_feature_key_retries_then_raises():
    client = FakeClient([MISSING_KEY_CODE])
    X = pd.DataFrame({"a": [0.0, 2.0]})
    with pytest.raises(ValueError, match="nonexistent"):
        generate_tree_with_retry(client=client, **_retry_kwargs(X=X, num_retry_llm=2))
    assert client.calls == 2


def test_check_complexity_param_accepts_numpy_integers():
    check_complexity_param(np.int64(2), "max_tree_depth")
    check_complexity_param(np.int32(3), "num_features")
    with pytest.raises(ValueError, match="max_tree_depth"):
        check_complexity_param(np.int64(0), "max_tree_depth")


class FailingClient:
    """Client whose two-step call itself fails (e.g. LLM API/network error)."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        raise self.exc


def test_client_errors_propagate_immediately_without_retry():
    # LLM call errors are not retried (only post-generation validation
    # failures trigger regeneration)
    client = FailingClient(RuntimeError("connection refused"))
    with pytest.raises(RuntimeError, match="connection refused"):
        generate_tree_with_retry(client=client, **_retry_kwargs(num_retry_llm=3))
    assert client.calls == 1


TRY_DEPTH_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "        try:\n"
    "            if nodes[1]:\n"
    "                prediction = 1\n"
    "        except Exception:\n"
    "            prediction = 2\n"
    "    else:\n"
    "        prediction = 3\n"
    "    return prediction, nodes\n"
)


def test_post_decision_overwrite_rejected_when_limit_configured():
    # nodes[0] is overwritten after the decision: the structure must be
    # unparsable, so a configured limit rejects and retries
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    nodes[0] = True\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None
    X = pd.DataFrame({"a": [0.0, 2.0]})
    client = FakeClient(["```python\n" + code + "```"])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(
            client=client,
            **_retry_kwargs(X=X, classes=None, numeric=True, max_tree_depth=1, num_retry_llm=2)
        )
    assert client.calls == 2


def test_prediction_overwrite_rejected_when_limit_configured():
    # the overwrite splits on b at both leaves (true decision depth 2), but
    # the stale structure parses as depth 1: max_tree_depth=1 must not be
    # bypassed by the unverified split
    code = (
        "def predict(X):\n"
        '    nodes = [X["a"] > 0]\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        '    prediction = 2 if X["b"] > 0 else 3\n'
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 0.0]})
    client = FakeClient(["```python\n" + code + "```"])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(
            client=client,
            **_retry_kwargs(X=X, classes=None, numeric=True, max_tree_depth=1, num_retry_llm=2)
        )
    assert client.calls == 2


def test_try_block_depth_enforced_by_max_tree_depth():
    # runtime depth is 2 (the second split is hidden inside a try block);
    # the structure must be unparsable so max_tree_depth=1 rejects and retries
    code = TRY_DEPTH_CODE
    assert parse_tree_function(code) is None
    X = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 2.0]})
    client = FakeClient(["```python\n" + code + "```"])
    with pytest.raises(ValueError, match="after 2 attempts"):
        generate_tree_with_retry(
            client=client,
            **_retry_kwargs(X=X, classes=None, numeric=True, max_tree_depth=1, num_retry_llm=2)
        )
    assert client.calls == 2
