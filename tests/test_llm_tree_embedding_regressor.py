"""Tests for llm_trees.LLMTreeEmbeddingRegressor (sklearn-compatible estimator)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from conftest import DIALOGUE_SEPARATOR, ScriptedClient, make_float_tree_client
from llm_trees._embeddings import downstream_preset
from llm_trees._llm_tree_embedding_regressor import LLMTreeEmbeddingRegressor
from llm_trees._tree import predict_rows

TARGET = {"y": "a continuous target"}

# features dict matching the X columns fitted in this file
# (the description text is irrelevant to these tests; only coverage is checked)
FD = {"a": "feature a", "b": "feature b"}


def make_X_y(n=20, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 2.0, n)
    b = rng.uniform(0.0, 3.0, n)
    X = pd.DataFrame({"a": a, "b": b})
    y = 2.0 * a + 3.0 * b + 0.1
    return X, y


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_regressor.make_client",
        lambda backend, temperature, seed: client,
    )


def embedding_estimator(**overrides):
    kwargs = dict(
        target=TARGET,
        features=FD,
        backend="openrouter",
        num_trees=2,
        estimator="lr",  # direct fit, no grid
        append_raw_features=False,
        prompt_file=False,
    )
    kwargs.update(overrides)
    return LLMTreeEmbeddingRegressor(**kwargs)


def test_fit_transform_predict_numeric(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = embedding_estimator().fit(X, y)
    T = est.transform(X)
    assert T.shape == (20, 2)  # 2 trees x 1 node
    assert T.dtype == np.float64
    pred = est.predict(X)
    assert pred.dtype == np.float64
    assert pred.shape == (20,)
    assert not hasattr(est, "classes_")


def test_transform_before_fit_raises():
    X, _ = make_X_y()
    est = embedding_estimator()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.transform(X)
    with pytest.raises(NotFittedError, match="not fitted yet"):
        est.predict(X)


def test_fit_llm_call_count_and_no_calls_after(monkeypatch):
    calls = {"n": 0, "seeds": []}

    def counting(backend, temperature, seed):
        calls["seeds"].append(seed)
        client = make_float_tree_client(["a", "b"])
        orig = client.two_step

        def two_step(a, b):
            calls["n"] += 1
            return orig(a, b)

        client.two_step = two_step
        return client

    monkeypatch.setattr("llm_trees._llm_tree_embedding_regressor.make_client", counting)
    X, y = make_X_y()
    est = embedding_estimator().fit(X, y)
    assert calls["seeds"] == [0, 1]
    assert calls["n"] == 2
    est.predict(X)
    est.transform(X)
    assert calls["n"] == 2  # no LLM calls after fit


def test_user_downstream_regressor(monkeypatch):
    class RecordingRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            self.X_fit_ = np.asarray(X)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = embedding_estimator(downstream_estimator=RecordingRegressor()).fit(X, y)
    assert isinstance(est.final_estimator_, Pipeline)
    model = est.final_estimator_.named_steps["model"]
    assert isinstance(model, RecordingRegressor)
    manual = np.hstack([predict_rows(fn, X)[1].astype(float) for _c, fn, _p in est.trees_])
    np.testing.assert_array_equal(model.X_fit_, manual)
    assert est.predict(X).shape == (20,)


def test_hgbdt_preset_end_to_end(monkeypatch):
    model, grid, scoring = downstream_preset("regressor", "hgbdt", 42)
    assert isinstance(model, HistGradientBoostingRegressor)
    assert scoring == "r2"
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = embedding_estimator(estimator="hgbdt").fit(X, y)
    assert isinstance(est.final_estimator_.named_steps["model"], HistGradientBoostingRegressor)
    assert est.predict(X).shape == (20,)


def test_fit_rejects_mismatched_sample_counts(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    with pytest.raises(ValueError, match="samples"):
        embedding_estimator(backend="openrouter").fit(X, y[:-3])


def test_invalid_regression_targets_rejected_before_provider_call(monkeypatch):
    X, y = make_X_y()
    calls = {"n": 0}

    def counting(backend, temperature, seed):
        calls["n"] += 1
        return make_float_tree_client(["a", "b"])

    monkeypatch.setattr("llm_trees._llm_tree_embedding_regressor.make_client", counting)
    with pytest.raises(ValueError, match="1d"):
        embedding_estimator(backend="openrouter").fit(X, y.reshape(-1, 1))
    with pytest.raises(ValueError, match="non-finite"):
        embedding_estimator(backend="openrouter").fit(X, np.array([1.0, float("nan")] * 10))
    with pytest.raises(ValueError, match="non-finite"):
        embedding_estimator(backend="openrouter").fit(X, np.array([1.0, float("inf")] * 10))
    with pytest.raises(ValueError, match="numeric"):
        embedding_estimator(backend="openrouter").fit(X, pd.Series(list("aabababababababababa")))
    assert calls["n"] == 0  # invalid targets never reach the provider


def test_invalid_params_and_target_rejected(monkeypatch):
    X, y = make_X_y()
    with pytest.raises(ValueError, match="num_trees"):
        embedding_estimator(num_trees=0).fit(X, y)
    with pytest.raises(ValueError, match="estimator"):
        embedding_estimator(estimator="xgb").fit(X, y)
    with pytest.raises(ValueError, match="target"):
        embedding_estimator(target={"y": ""}).fit(X, y)
    # non-numeric target
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    with pytest.raises(ValueError, match="numeric"):
        embedding_estimator().fit(X, pd.Series(list("aabababababababababa")))
    # 2-d target
    with pytest.raises(ValueError, match="1d"):
        embedding_estimator().fit(X, y.reshape(-1, 1))


def test_append_raw_features_dimension(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = embedding_estimator(append_raw_features=True).fit(X, y)
    assert est.transform(X).shape == (20, 2 + 2)  # nodes + 2 scaled raw


def test_numpy_input_with_features(monkeypatch):
    # the features dict keys define the column names for NumPy input
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = embedding_estimator().fit(X.to_numpy(), y)
    assert est.feature_names_in_.tolist() == ["a", "b"]
    assert est.predict(X.to_numpy()).shape == (20,)


def test_clone_get_set_params():
    est = LLMTreeEmbeddingRegressor(
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

def _raw(fenced_code: str) -> str:
    return "A decision tree.\n\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + fenced_code


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
    y = (2.0 * X["a"] + X["b"]).to_numpy()
    return X, y


def _fit_width_tree(monkeypatch, code):
    X_fit, y_fit = _width_fit_X_y()
    _patch_client(monkeypatch, ScriptedClient(_raw(code)))
    est = embedding_estimator(num_trees=1).fit(X_fit, y_fit)
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

