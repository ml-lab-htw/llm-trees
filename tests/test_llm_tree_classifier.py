"""Tests for llm_trees.LLMTreeClassifier (sklearn-compatible estimator)."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

from llm_trees import _llm
from llm_trees._llm_tree_classifier import LLMTreeClassifier
from llm_trees._tree import DIALOGUE_SEPARATOR

TARGET = {"x1": "synthetic system state (normal / abnormal)"}

# features dicts matching the X column sets used in this file
# (the description text is irrelevant to these tests; only coverage is checked)
FD = {"a": "feature a", "b": "feature b"}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        return self.responses[min(self.calls, len(self.responses)) - 1]


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(
        "llm_trees._llm_tree_classifier.make_client",
        lambda backend, temperature, seed: client,
    )


def test_predict_numeric_labels_preserve_dtype_and_score(monkeypatch):
    numeric_tree_code = (
        "```python\n"
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    return prediction, nodes\n"
        "```"
    )
    X = pd.DataFrame({"a": [0.0, 2.0, 0.5, 1.5, 0.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    _patch_client(monkeypatch, FakeClient([numeric_tree_code]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    pred = est.predict(X)
    assert pred.dtype == np.int64  # dtype preserved from classes_
    np.testing.assert_array_equal(pred, [0, 1, 0, 1, 0, 1])
    # sklearn scoring works on numeric labels
    assert accuracy_score(y, pred) == 1.0
    assert est.score(X, y) == 1.0


def test_predict_before_fit_raises():
    X, _ = small_X_y()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        LLMTreeClassifier(target=TARGET, features=FD).predict(X)


def test_predict_proba_empty_leaf_uses_prior(monkeypatch):
    deep_code = (
        "```python\n"
        "def predict(X: dict):\n"
        "    nodes = 2 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        '    nodes[1] = X["b"] <= 2.0\n'
        "    if nodes[0]:\n"
        '        prediction = "normal"\n'
        "    else:\n"
        "        if nodes[1]:\n"
        '            prediction = "abnormal"\n'
        "        else:\n"
        '            prediction = "normal"\n'
        "    return prediction, nodes\n"
        "```"
    )
    # fit rows fill leaf 0 (all "normal") and leaf 1 (all "abnormal"); leaf 2 stays empty
    X_fit = pd.DataFrame({"a": [0.0] * 10 + [2.0] * 10, "b": [0.0] * 10 + [1.0] * 10})
    y_fit = pd.Series(["normal"] * 10 + ["abnormal"] * 10)
    _patch_client(monkeypatch, FakeClient([deep_code]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X_fit, y_fit)

    # a row that lands in the empty leaf 2 gets the class prior
    proba = est.predict_proba(pd.DataFrame({"a": [2.0], "b": [3.0]}))
    assert proba.shape == (1, 2)
    np.testing.assert_allclose(proba[0], [0.5, 0.5])  # classes_: ["abnormal", "normal"]
    # fit rows get the (pure) leaf fractions
    fit_proba = est.predict_proba(X_fit)
    np.testing.assert_allclose(fit_proba[:10], [[0.0, 1.0]] * 10)
    np.testing.assert_allclose(fit_proba[10:], [[1.0, 0.0]] * 10)


def test_predict_proba_unparsable_tree_raises(monkeypatch):
    unparsable = (
        "```python\n"
        "def predict(X: dict):\n"
        '    prediction = "normal" if X["a"] < 1.0 else "abnormal"\n'
        "    return prediction, [True]\n"
        "```"
    )
    X = pd.DataFrame({"a": [0.0, 2.0]})
    y = pd.Series(["normal", "abnormal"])
    _patch_client(monkeypatch, FakeClient([unparsable]))
    est = LLMTreeClassifier(target=TARGET, features={"a": "feature a"}, backend="openrouter").fit(X, y)
    assert est.tree_ is None
    with pytest.raises(RuntimeError, match="parsable tree structure"):
        est.predict_proba(X)
    # predict still works
    assert set(np.unique(est.predict(X))) == {"normal", "abnormal"}


def test_predict_no_llm_calls_and_no_files(tmp_path, monkeypatch):
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(_llm, "OpenAI") as mock_openai:
        est = LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
        est.predict(X)
        est.predict_proba(X)
    mock_openai.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_numpy_input_with_features(monkeypatch):
    # the features dict keys define the column names for NumPy input
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X.to_numpy(), y)
    assert est.feature_names_in_.tolist() == ["a", "b"]
    assert est.predict(X.to_numpy()).shape == (6,)


def test_numpy_input_feature_count_mismatch_rejected():
    X, y = small_X_y()
    with pytest.raises(ValueError, match="features has 1 entries, but X has 2 features"):
        LLMTreeClassifier(target=TARGET, features={"a": "feature a"}).fit(X.to_numpy(), y)


def test_fit_rejects_mismatched_sample_counts():
    X, y = small_X_y()
    with pytest.raises(ValueError, match="samples"):
        LLMTreeClassifier(target=TARGET, features=FD).fit(X, y.iloc[:5])
    with pytest.raises(ValueError, match="samples"):
        LLMTreeClassifier(target=TARGET, features=FD).fit(X, pd.concat([y, y.iloc[:5]]))


def test_numpy_integer_complexity_params_accepted(monkeypatch):
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    est = LLMTreeClassifier(
        target=TARGET, features=FD, backend="openrouter", max_tree_depth=np.int64(2), num_features=np.int64(3)
    ).fit(X, y)
    assert est.sparsity_ <= 3


def test_missing_openrouter_env_vars_raise_immediately(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    X, y = small_X_y()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMTreeClassifier(target=TARGET, features=FD).fit(X, y)


GOOD_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "normal"\n'
    "    else:\n"
    '        prediction = "abnormal"\n'
    "    return prediction, nodes\n"
    "```"
)


class AlwaysGarbageClient:
    def __init__(self):
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        return "no code at all in this response"


def small_X_y():
    X = pd.DataFrame({"a": [0.0, 2.0, 0.5, 1.5, 0.0, 2.5], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series(["normal", "abnormal", "normal", "abnormal", "normal", "abnormal"])
    return X, y


def test_failed_initial_fit_leaves_estimator_unfitted(monkeypatch):
    X, y = small_X_y()
    client = AlwaysGarbageClient()
    _patch_client(monkeypatch, client)
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter", num_retry_llm=2)
    with pytest.raises(ValueError, match="after 2 attempts"):
        est.fit(X, y)
    # no partial fitted state: inference must raise NotFittedError, not AttributeError
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.predict(X)
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.predict_proba(X)


def test_failed_refit_keeps_previous_state(monkeypatch):
    X, y_old = small_X_y()
    y_new = pd.Series(["c", "d", "c", "d", "c", "d"])
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y_old)
    old_pred = est.predict(X)
    assert set(old_pred.tolist()) <= {"normal", "abnormal"}

    # a refit with different labels that fails must not mix old tree with new classes
    _patch_client(monkeypatch, AlwaysGarbageClient())
    with pytest.raises(ValueError, match="after 10 attempts"):
        est.fit(X, y_new)
    np.testing.assert_array_equal(est.predict(X), old_pred)
    assert est.classes_.tolist() == ["abnormal", "normal"]


CONSTANT_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 0 * [None]\n"
    '    prediction = "normal"\n'
    "    return prediction, nodes\n"
    "```"
)


def test_constant_tree_with_complexity_limits_accepted(monkeypatch):
    # a single-leaf (constant) tree satisfies any depth/feature limit
    # (depth 0, 0 features); a parser that rejects it sends a model that
    # keeps generating it into paid retries until abort
    X, y = small_X_y()
    raw = "A constant tree." + "\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + CONSTANT_CLS_CODE
    client = FakeClient([raw])
    _patch_client(monkeypatch, client)
    est = LLMTreeClassifier(
        target=TARGET, features=FD, backend="openrouter",
        max_tree_depth=1, num_features=1,
    ).fit(X, y)
    assert client.calls == 1  # accepted on the first attempt
    np.testing.assert_array_equal(est.predict(X), np.array(["normal"] * 6))
    # every sample lands in the single leaf: proba is the fit class prior
    proba = est.predict_proba(X)
    np.testing.assert_allclose(proba, np.full((6, 2), 0.5))
    assert est.tree_ is not None and est.tree_.depth == 0
    assert est.sparsity_ == 0 and len(est.support_) == 0


NUM_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        prediction = 1\n"
    "    return prediction, nodes\n"
    "```\n"
)


def test_single_column_targets_normalized(monkeypatch):
    X, _ = small_X_y()
    _patch_client(monkeypatch, FakeClient([NUM_CLS_CODE]))
    # (n, 1) array
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, np.array([[0], [1]] * 3))
    assert est.classes_.tolist() == [0, 1]
    assert est.predict(X).dtype == np.int64
    # one-column DataFrame (its column name must match the target key)
    est2 = LLMTreeClassifier(target={"y": "t"}, features=FD, backend="openrouter").fit(
        X, pd.DataFrame({"y": [0, 1, 0, 1, 0, 1]})
    )
    assert est2.classes_.tolist() == [0, 1]
    assert est2.predict(X).shape == (6,)
    # plain 1d labels keep working
    est3 = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, np.array([0, 1, 0, 1, 0, 1]))
    assert est3.predict(X).shape == (6,)


def test_multioutput_classification_target_rejected_before_provider(monkeypatch):
    X, _ = small_X_y()
    calls = {"n": 0}

    def counting(backend, temperature, seed):
        calls["n"] += 1
        return FakeClient([GOOD_CLS_CODE])

    monkeypatch.setattr("llm_trees._llm_tree_classifier.make_client", counting)
    with pytest.raises(ValueError, match="1d"):
        LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(
            X, np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
        )
    with pytest.raises(ValueError, match="1d"):
        LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(
            X, pd.DataFrame({"y1": [0, 1, 0, 1, 0, 1], "y2": [1, 0, 1, 0, 1, 0]})
        )
    assert calls["n"] == 0  # multioutput never reaches the provider


def test_duplicate_column_names_rejected_before_provider(monkeypatch):
    # shared input validation (resolve_feature_names) must reject duplicate
    # DataFrame column names before any prompt is built or provider called
    X, y = small_X_y()
    X = X.copy()
    X.columns = ["a", "a"]
    calls = {"n": 0}

    def counting(backend, temperature, seed):
        calls["n"] += 1
        return FakeClient([GOOD_CLS_CODE])

    monkeypatch.setattr("llm_trees._llm_tree_classifier.make_client", counting)
    with pytest.raises(ValueError, match="duplicate"):
        LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert calls["n"] == 0  # duplicate columns never reach the provider


TRY_LEAF_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "abnormal"\n'
    "    else:\n"
    '        prediction = "normal"\n'
    "        try:\n"
    '            prediction = "abnormal"\n'
    "        except Exception:\n"
    "            pass\n"
    "    return prediction, nodes\n"
    "```\n"
)


def test_unverifiable_control_flow_fallback_without_limits(monkeypatch):
    # a try branch overwrites the visible else-leaf at runtime; without
    # configured limits the documented fallback applies: the tree fits and
    # predicts via the compiled function, but its structure is untrusted
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([TRY_LEAF_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert est.tree_ is None
    assert len(est.support_) == 0
    # runtime leaf values: if-leaf "abnormal"; else-leaf overwritten to "abnormal"
    assert set(np.unique(est.predict(X)).tolist()) == {"abnormal"}
    # leaf probabilities cannot be trusted for an unparsable structure
    with pytest.raises(RuntimeError, match="parsable"):
        est.predict_proba(X)


MUTATE_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        if nodes[1]:\n"
    '            prediction = "normal"\n'
    "        else:\n"
    '            prediction = "abnormal"\n'
    "    else:\n"
    "        nodes[1] = nodes.reverse()\n"
    '        prediction = "normal"\n'
    "    return prediction, nodes\n"
    "```\n"
)


def test_value_side_node_mutation_retries_with_limits(monkeypatch):
    # nodes.reverse() in the assignment value mutates the container after the
    # nodes[0] decision (the returned vector no longer reproduces the
    # decisions); the predictions stay mappable and the node widths stay
    # consistent, so only the structural parse can reject the tree
    X, y = small_X_y()
    client = FakeClient([MUTATE_CLS_CODE])
    _patch_client(monkeypatch, client)
    with pytest.raises(ValueError, match="after 2 attempts"):
        LLMTreeClassifier(
            target=TARGET, features=FD, backend="openrouter",
            num_retry_llm=2, max_tree_depth=2,
        ).fit(X, y)
    assert client.calls == 2


def test_value_side_node_mutation_fallback_without_limits(monkeypatch):
    # without configured limits the documented fallback applies: the tree
    # fits and predicts via the compiled function, but its structure is
    # untrusted (leaf probabilities would silently mix runtime leaves)
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([MUTATE_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert est.tree_ is None
    assert len(est.support_) == 0
    assert set(np.unique(est.predict(X)).tolist()) == {"normal", "abnormal"}
    with pytest.raises(RuntimeError, match="parsable"):
        est.predict_proba(X)


OVERWRITE_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "abnormal"\n'
    "    else:\n"
    '        prediction = "normal"\n'
    "    nodes[0] = True\n"
    "    return prediction, nodes\n"
    "```\n"
)


def test_post_decision_overwrite_fallback_without_limits(monkeypatch):
    # the returned nodes no longer reproduce the decisions (both leaves
    # would walk to the same parsed leaf); without limits the documented
    # fallback applies: prediction works via the compiled function, but the
    # untrusted structure gives an empty support_ and no leaf probabilities
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([OVERWRITE_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert set(np.unique(est.predict(X)).tolist()) == {"normal", "abnormal"}
    assert est.tree_ is None
    assert len(est.support_) == 0
    with pytest.raises(RuntimeError, match="parsable"):
        est.predict_proba(X)


OVERWRITE_PRED_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "abnormal"\n'
    "    else:\n"
    '        prediction = "normal"\n'
    '    prediction = "normal" if X["b"] > 1.5 else "abnormal"\n'
    "    return prediction, nodes\n"
    "```\n"
)


def test_prediction_overwrite_fallback_without_limits(monkeypatch):
    # the executed prediction overwrites both leaves with a b-based value:
    # the stale structure is untrustworthy, so without limits the documented
    # fallback applies: prediction works via the compiled function, support_
    # stays empty, and predict_proba raises instead of returning wrong
    # leaf fractions
    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([OVERWRITE_PRED_CLS_CODE]))
    est = LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert set(np.unique(est.predict(X)).tolist()) == {"normal", "abnormal"}
    assert est.tree_ is None
    assert len(est.support_) == 0
    with pytest.raises(RuntimeError, match="parsable"):
        est.predict_proba(X)


def test_fit_unknown_feature_key_raises(monkeypatch):
    typo_code = (
        "```python\n"
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["typo"] <= 1.0\n'
        "    if nodes[0]:\n"
        '        prediction = "normal"\n'
        "    else:\n"
        '        prediction = "abnormal"\n'
        "    return prediction, nodes\n"
        "```"
    )
    X, y = small_X_y()
    client = FakeClient([typo_code])
    _patch_client(monkeypatch, client)
    with pytest.raises(ValueError, match="typo"):
        LLMTreeClassifier(
            target=TARGET, features=FD, backend="openrouter", num_retry_llm=2
        ).fit(X, y)
    assert client.calls == 2


def test_type_mismatch_surfaces_clearly(monkeypatch):
    type_code = (
        "```python\n"
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 170\n'
        "    if nodes[0]:\n"
        '        prediction = "normal"\n'
        "    else:\n"
        '        prediction = "abnormal"\n'
        "    return prediction, nodes\n"
        "```"
    )
    X = pd.DataFrame({"a": ["165.5", "180.2"], "b": [1.0, 2.0]})
    y = pd.Series(["normal", "abnormal"])
    client = FakeClient([type_code])
    _patch_client(monkeypatch, client)
    with pytest.raises(ValueError, match="not supported between"):
        LLMTreeClassifier(
            target=TARGET,
            features={"a": "value of a (0 = low, 1 = high)", "b": "value of b in millimeters"},
            backend="openrouter",
            num_retry_llm=2,
        ).fit(X, y)
    assert client.calls == 2


@pytest.mark.parametrize(
    "params",
    [
        {"max_tree_depth": 0},
        {"max_tree_depth": -1},
        {"max_tree_depth": 2.5},
        {"max_tree_depth": True},
        {"num_features": 0},
        {"num_features": "3"},
    ],
)
def test_invalid_complexity_params_rejected(params):
    X, y = small_X_y()
    with pytest.raises(ValueError, match="max_tree_depth|num_features"):
        LLMTreeClassifier(target=TARGET, features=FD, **params).fit(X, y)


@pytest.mark.parametrize(
    "target",
    ["t", 3, {}, {"a": "1", "b": "2"}, {"x1": "  "}, {"x1": None}, {None: "t"}, {3: "t"}],
)
def test_invalid_target_rejected(target):
    X, y = small_X_y()
    with pytest.raises(ValueError, match="target"):
        LLMTreeClassifier(target=target, features=FD).fit(X, y)


@pytest.mark.parametrize(
    "features",
    ["a, b", 3, {}, {"a": "  ", "b": "feature b"}, {"a": None, "b": "feature b"}, {None: "d", "b": "d"}],
)
def test_invalid_features_rejected(features):
    X, y = small_X_y()
    with pytest.raises(ValueError, match="feature"):
        LLMTreeClassifier(target=TARGET, features=features).fit(X, y)


def test_features_keys_must_match_X_columns():
    X, y = small_X_y()
    with pytest.raises(ValueError, match="missing: \\['b'\\]"):
        LLMTreeClassifier(target=TARGET, features={"a": "feature a"}).fit(X, y)
    with pytest.raises(ValueError, match="unseen: \\['c'\\]"):
        LLMTreeClassifier(target=TARGET, features={"a": "feature a", "b": "feature b", "c": "feature c"}).fit(
            X, y
        )


def test_target_key_must_match_named_y(monkeypatch):
    X = small_X_y()[0]
    y = pd.Series(["normal", "abnormal", "normal", "abnormal", "normal", "abnormal"], name="x1")
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    with pytest.raises(ValueError, match="does not match the y column name"):
        LLMTreeClassifier(target={"other": "t"}, features=FD, backend="openrouter").fit(X, y)
    # matching name: fit proceeds
    LLMTreeClassifier(target={"x1": "t"}, features=FD, backend="openrouter").fit(X, y)
    # one-column DataFrame y carries its column name
    with pytest.raises(ValueError, match="does not match the y column name"):
        LLMTreeClassifier(target={"other": "t"}, features=FD, backend="openrouter").fit(
            X, pd.DataFrame({"y": [0, 1, 0, 1, 0, 1]})
        )


def test_clone_get_set_params():
    est = LLMTreeClassifier(
        target=TARGET,
        features={"a": "d"},
        max_tree_depth=2,
        num_features=3,
        prompt_file="p.txt",
        backend="openrouter",
        temperature=0.5,
        num_examples=2,
        use_role_prompt=True,
        num_retry_llm=7,
        seed=1,
    )
    c = clone(est)
    assert c.get_params() == est.get_params()
    est.set_params(num_retry_llm=5)
    assert est.get_params()["num_retry_llm"] == 5


def test_first_prompt_sees_target_and_feature_descriptions(monkeypatch):
    # the tree-induction prompt must carry the target and each feature's
    # name + description, not just cryptic column keys
    import llm_trees._llm_tree_classifier as mod

    X, y = small_X_y()
    _patch_client(monkeypatch, FakeClient([GOOD_CLS_CODE]))
    calls = []
    real_build = mod.build_first_prompt

    def spy(**kwargs):
        calls.append(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(mod, "build_first_prompt", spy)
    LLMTreeClassifier(
        target={"x1": "synthetic system state (normal / abnormal)"},
        features={
            "a": "value of a (0 = low, 1 = high)",
            "b": "value of b in millimeters",
        },
        backend="openrouter",
    ).fit(X, y)
    assert calls, "build_first_prompt was not called"
    prompt = real_build(**calls[0])
    assert "The tree should predict the synthetic system state (normal / abnormal)." in prompt
    assert "Features: a: value of a (0 = low, 1 = high), b: value of b in millimeters" in prompt


def test_target_and_features_are_required():
    X, y = small_X_y()
    with pytest.raises(TypeError, match="target"):
        LLMTreeClassifier(features=FD).fit(X, y)
    with pytest.raises(TypeError, match="features"):
        LLMTreeClassifier(target=TARGET).fit(X, y)
