"""Tests for llm_trees.LLMTreeEmbeddingClassifier (sklearn-compatible estimator)."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from conftest import DIALOGUE_SEPARATOR, ScriptedClient
from llm_trees import _llm
from llm_trees._llm_tree_embedding_classifier import LLMTreeEmbeddingClassifier

TARGET = {"x1": "synthetic system state (normal / abnormal)"}

# features dict matching the X column sets used in this file
# (the description text is irrelevant to these tests; only coverage is checked)
FD = {"a": "feature a", "b": "feature b"}

# A synthetic depth-2 tree on the two synthetic features: 3 nodes, used as
# the scripted step-2 response for tests that fit real estimators.
DEPTH2_CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 3 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    '    nodes[2] = X["a"] <= 0.5\n'
    "    if nodes[0]:\n"
    "        if nodes[1]:\n"
    '            prediction = "normal"\n'
    "        else:\n"
    '            prediction = "abnormal"\n'
    "    else:\n"
    '        prediction = "abnormal"\n'
    "    return prediction, nodes\n"
    "```"
)


def embedding_estimator(**overrides):
    kwargs = dict(
        target=TARGET,
        features=FD,
        max_tree_depth=2,
        backend="openrouter",
        num_trees=2,
        estimator="lr",  # fast preset for tests
        append_raw_features=False,
        prompt_file=False,
    )
    kwargs.update(overrides)
    return LLMTreeEmbeddingClassifier(**kwargs)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.X_fit_ = np.asarray(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array(["normal"] * len(X))


def test_fit_transform_predict_shapes(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    est = embedding_estimator().fit(X, y)
    T = est.transform(X)
    assert T.shape == (12, 6)  # 2 trees x 3 nodes
    assert T.dtype == np.float64
    pred = est.predict(X)
    assert pred.shape == (12,)
    assert set(np.unique(pred)) <= set(y.unique())


def test_transform_train_inference_dimension_consistency(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    est = embedding_estimator().fit(X, y)
    T_train = est.transform(X.head(7))
    T_test = est.transform(X.tail(5))
    assert T_train.shape[1] == T_test.shape[1] == 6


def test_append_raw_features_dimension(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    est = embedding_estimator(append_raw_features=True).fit(X, y)
    T = est.transform(X)
    assert T.shape == (12, 6 + 2)  # nodes + 2 scaled numeric raw features


def test_fit_makes_num_trees_llm_calls_and_no_calls_after(monkeypatch):
    calls = {"seeds": []}
    client = ScriptedClient(_raw(DEPTH2_CLS_CODE))

    def counting(backend, temperature, seed):
        calls["seeds"].append(seed)
        return client

    monkeypatch.setattr("llm_trees._llm_tree_embedding_classifier.make_client", counting)
    X, y = _small_X_y()
    est = embedding_estimator().fit(X, y)
    # per-tree seed: induction of tree i uses seed i
    assert calls["seeds"] == [0, 1]
    assert client.calls == 2
    est.predict(X)
    est.transform(X)
    assert client.calls == 2  # no LLM calls after fit


def test_predict_transform_before_fit_raise():
    X, _ = _small_X_y()
    est = embedding_estimator()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.predict(X)
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.transform(X)


def test_fitted_attributes_union_support(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    est = embedding_estimator().fit(X, y)
    assert len(est.trees_) == 2
    for code, fn, pt in est.trees_:
        assert isinstance(code, str) and callable(fn)
    assert est.support_.tolist() == ["a", "b"]
    assert est.sparsity_ == 2
    assert set(est.classes_) == {"normal", "abnormal"}
    assert est.elapsed_time_ >= 0
    assert est.feature_names_in_.tolist() == ["a", "b"]
    assert isinstance(est.final_estimator_, Pipeline)


def test_downstream_estimator_bypasses_grid(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    est = embedding_estimator(downstream_estimator=RecordingClassifier()).fit(X, y)
    model = est.final_estimator_.named_steps["model"]
    assert isinstance(model, RecordingClassifier)
    # captured embedding equals the manual node-matrix hstack
    from llm_trees._tree import predict_rows

    manual = np.hstack([predict_rows(fn, X)[1].astype(float) for _c, fn, _p in est.trees_])
    np.testing.assert_array_equal(model.X_fit_, manual)
    assert est.predict(X).shape == (12,)


def test_fit_rejects_mismatched_sample_counts(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    with pytest.raises(ValueError, match="samples"):
        embedding_estimator().fit(X, y.iloc[:5])
    with pytest.raises(ValueError, match="samples"):
        embedding_estimator().fit(X, pd.concat([y, y.iloc[:5]]))


def _raw(fenced_code: str) -> str:
    return "A decision tree.\n\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + fenced_code


def _small_X_y():
    # 6 samples per class: the "lr" preset grid-searches with cv=3, which
    # requires at least 3 members in each class
    X = pd.DataFrame(
        {
            "a": [0.0, 2.0, 0.5, 1.5, 0.2, 1.8] * 2,
            "b": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5] * 2,
        }
    )
    y = pd.Series(["normal", "abnormal"] * 6)
    return X, y


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_classifier.make_client",
        lambda backend, temperature, seed: client,
    )


_GOOD_CLS_CODE = (
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


class GarbageClient:
    def __init__(self):
        self.calls = 0

    def two_step(self, first_prompt, second_prompt):
        self.calls += 1
        return "no code at all in this response"


def test_fit_rejects_invalid_node_vectors(monkeypatch):
    nodes_none = _raw(
        "```python\n"
        "def predict(X: dict):\n"
        '    prediction = 0 if X["a"] < 1.0 else 1\n'
        "    return prediction, None\n"
        "```"
    )
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(nodes_none))
    with pytest.raises(ValueError, match="nodes"):
        LLMTreeEmbeddingClassifier(
            target=TARGET, features=FD, backend="openrouter", num_trees=1, num_retry_llm=2,
            append_raw_features=False,
        ).fit(X, y)


def test_fit_rejects_dict_node_vectors_before_downstream_fit(monkeypatch):
    # a dict node vector must be rejected in the retry/validation phase and
    # never reach the embedding construction (where NumPy would convert it
    # silently, e.g. {0: True, 1: False} -> a constant column pair)
    dict_nodes = _raw(
        "```python\n"
        "def predict(X: dict):\n"
        "    nodes = {0: True, 1: False}\n"
        '    prediction = "normal"\n'
        "    return prediction, nodes\n"
        "```"
    )
    X, y = _small_X_y()
    client = ScriptedClient(dict_nodes)
    _patch_client(monkeypatch, client)
    est = LLMTreeEmbeddingClassifier(
        target=TARGET, features=FD, backend="openrouter", num_trees=1, num_retry_llm=2,
        append_raw_features=False,
    )
    with pytest.raises(ValueError, match="nodes"):
        est.fit(X, y)
    assert client.calls == 2  # retried, not accepted
    # fit failed before any downstream state was committed
    with pytest.raises(NotFittedError):
        est.predict(X)


def test_failed_refit_keeps_previous_state(monkeypatch):
    X, y_old = _small_X_y()
    y_new = pd.Series(["c", "d"] * 6)
    _patch_client(monkeypatch, ScriptedClient(_raw(_GOOD_CLS_CODE)))
    est = embedding_estimator(features=FD, backend="openrouter", num_trees=1).fit(X, y_old)
    old_pred = est.predict(X)

    _patch_client(monkeypatch, GarbageClient())
    with pytest.raises(ValueError, match="attempts"):
        est.fit(X, y_new)
    # old state stays intact: no mixed old-tree/new-classes predictions
    np.testing.assert_array_equal(est.predict(X), old_pred)
    assert est.classes_.tolist() == ["abnormal", "normal"]


_NUM_CLS_CODE = (
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


def test_single_column_and_multioutput_targets(monkeypatch):
    X, _ = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(_NUM_CLS_CODE)))
    # (n, 1) array target is normalized to 1d
    est = embedding_estimator(features=FD, backend="openrouter", num_trees=1).fit(X, np.array([[0], [1]] * 6))
    assert est.classes_.tolist() == [0, 1]
    assert est.predict(X).shape == (12,)

    # multioutput is rejected before any tree generation
    _patch_client(monkeypatch, GarbageClient())
    with pytest.raises(ValueError, match="1d"):
        embedding_estimator(features=FD, backend="openrouter", num_trees=1).fit(
            X, np.array([[0, 1], [1, 0]] * 6)
        )


def test_invalid_params_rejected(monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    with pytest.raises(ValueError, match="num_trees"):
        embedding_estimator(num_trees=0).fit(X, y)
    with pytest.raises(ValueError, match="num_trees"):
        embedding_estimator(num_trees=True).fit(X, y)
    with pytest.raises(ValueError, match="estimator"):
        embedding_estimator(estimator="xgb").fit(X, y)
    with pytest.raises(ValueError, match="max_tree_depth"):
        embedding_estimator(max_tree_depth=0).fit(X, y)
    with pytest.raises(ValueError, match="target"):
        embedding_estimator(target={"x1": ""}).fit(X, y)


def test_no_files_created(tmp_path, monkeypatch):
    X, y = _small_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(DEPTH2_CLS_CODE)))
    with mock.patch.object(_llm, "OpenAI") as mock_openai:
        est = embedding_estimator().fit(X, y)
        est.transform(X)
        est.predict(X)
    mock_openai.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_clone_get_set_params():
    est = LLMTreeEmbeddingClassifier(
        target=TARGET,
        features={"a": "d"},
        max_tree_depth=2,
        num_features=3,
        num_trees=4,
        append_raw_features=False,
        estimator="hgbdt",
        categorical_features=["a"],
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
    est.set_params(num_trees=3)
    assert est.get_params()["num_trees"] == 3


# --- per-row node-width validation at inference time -------------------------

LATER_WIDTH1_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    if X["a"] <= 1.0:\n'
    '        return 0, [X["a"] <= 1.0, X["b"] <= 2.0]\n'
    "    else:\n"
    "        return 1, [True]\n"
    "```\n"
)

LATER_WIDTH3_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    if X["a"] <= 1.0:\n'
    '        return 0, [X["a"] <= 1.0, X["b"] <= 2.0]\n'
    "    else:\n"
    "        return 1, [True, False, True]\n"
    "```\n"
)


def _width_fit_X_y():
    # every fit row takes the width-2 branch (a <= 1.0)
    X = pd.DataFrame(
        {
            "a": [0.2, 0.8, 0.4, 0.6, 0.3, 0.9] * 2,
            "b": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5] * 2,
        }
    )
    y = pd.Series([0, 1] * 6)
    return X, y


def _fit_width_tree(monkeypatch, code):
    X_fit, y_fit = _width_fit_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(code)))
    est = embedding_estimator(
        features=FD,
        backend="openrouter",
        num_trees=1,
        estimator="lr",
        append_raw_features=False,
        max_tree_depth=None,  # unparsable tree: the documented fallback applies
    ).fit(X_fit, y_fit)
    return est, X_fit


def test_later_row_width_mismatch_rejected(monkeypatch):
    # training rows width 2; first inference row width 2; a later unseen
    # branch returns width 1 -> must be rejected, not broadcast silently
    est, X_fit = _fit_width_tree(monkeypatch, LATER_WIDTH1_CODE)
    assert est.transform(X_fit).shape == (12, 2)  # valid training embedding
    X_bad = pd.DataFrame({"a": [0.5, 5.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="width"):
        est.transform(X_bad)
    with pytest.raises(ValueError, match="width"):
        est.predict(X_bad)


def test_later_row_width_too_long_rejected(monkeypatch):
    est, _ = _fit_width_tree(monkeypatch, LATER_WIDTH3_CODE)
    X_bad = pd.DataFrame({"a": [0.5, 5.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="width"):
        est.transform(X_bad)
    with pytest.raises(ValueError, match="width"):
        est.predict(X_bad)


def test_first_row_width_mismatch_rejected(monkeypatch):
    est, _ = _fit_width_tree(monkeypatch, LATER_WIDTH1_CODE)
    X_bad = pd.DataFrame({"a": [5.0], "b": [2.0]})
    with pytest.raises(ValueError, match="width"):
        est.transform(X_bad)
    with pytest.raises(ValueError, match="width"):
        est.predict(X_bad)


