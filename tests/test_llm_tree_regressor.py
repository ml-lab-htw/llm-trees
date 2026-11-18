"""Tests for llm_trees.LLMTreeRegressor (sklearn-compatible estimator)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import make_float_tree_client
from llm_trees._llm_tree_regressor import LLMTreeRegressor

TARGET = {"y": "a continuous target"}

# features dict matching the X columns fitted in this file
# (the description text is irrelevant to these tests; only coverage is checked)
FD = {"a": "feature a", "b": "feature b"}

STR_TREE_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    prediction = "oops"\n'
    "    return prediction, [True]\n"
    "```"
)


def make_X_y():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 2.0, 30)
    b = rng.uniform(0.0, 3.0, 30)
    X = pd.DataFrame({"a": a, "b": b})
    y = 2.0 * a + 3.0 * b + 0.1
    return X, y


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(
        "llm_trees._llm_tree_regressor.make_client",
        lambda backend, temperature, seed: client,
    )


def test_fit_predict_numeric_shape(monkeypatch):
    X, y = make_X_y()
    client = make_float_tree_client(["a", "b"])
    _patch_client(monkeypatch, client)
    est = LLMTreeRegressor(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    pred = est.predict(X)
    assert pred.dtype == np.float64
    assert pred.shape == (30,)
    np.testing.assert_allclose(pred, 2.0 * X["a"].to_numpy() + X["b"].to_numpy())
    assert client.calls == 1


def test_fitted_attributes(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = LLMTreeRegressor(target=TARGET, features=FD, backend="openrouter").fit(X, y)
    assert est.feature_names_in_.tolist() == ["a", "b"]
    assert est.elapsed_time_ >= 0
    assert isinstance(est.tree_code_, str) and "def predict" in est.tree_code_
    assert est.tree_ is None  # plain-return form is not parsable
    assert not hasattr(est, "classes_")
    assert not hasattr(est, "predict_proba")


def test_predict_before_fit_raises():
    X, _ = make_X_y()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        LLMTreeRegressor(target=TARGET, features=FD).predict(X)


def test_non_numeric_tree_output_retries_then_raises(monkeypatch):
    X, y = make_X_y()
    client = make_float_tree_client(["a", "b"])
    client.raw_response = STR_TREE_CODE
    _patch_client(monkeypatch, client)
    with pytest.raises(ValueError, match="after 2 attempts"):
        LLMTreeRegressor(
            target=TARGET, features=FD, backend="openrouter", num_retry_llm=2
        ).fit(X, y)
    assert client.calls == 2


def test_fit_rejects_mismatched_sample_counts(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    with pytest.raises(ValueError, match="samples"):
        LLMTreeRegressor(target=TARGET, features=FD, backend="openrouter").fit(X, y[:-3])


def test_invalid_regression_targets_rejected_before_provider_call(monkeypatch):
    X, y = make_X_y()
    calls = {"n": 0}

    def counting(backend, temperature, seed):
        calls["n"] += 1
        return make_float_tree_client(["a", "b"])

    monkeypatch.setattr("llm_trees._llm_tree_regressor.make_client", counting)
    with pytest.raises(ValueError, match="1d"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, y.reshape(-1, 1))
    with pytest.raises(ValueError, match="non-finite"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, np.array([1.0, float("nan"), 3.0] * 10))
    with pytest.raises(ValueError, match="non-finite"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, np.array([1.0, float("inf"), 3.0] * 10))
    with pytest.raises(ValueError, match="non-finite"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, np.array([1.0, float("-inf"), 3.0] * 10))
    with pytest.raises(ValueError, match="numeric"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, pd.Series(list("aabab" * 6)))
    # complex targets: casting to float64 would silently discard the
    # imaginary part, so they must be rejected before conversion
    with pytest.raises(ValueError, match="complex"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 0.0j] * 10))
    with pytest.raises(ValueError, match="complex"):
        LLMTreeRegressor(target=TARGET, features=FD).fit(X, np.array([1.0 + float("inf") * 1j, 3.0, 5.0] * 10))
    assert calls["n"] == 0  # invalid targets never reach the provider


def test_valid_regression_targets_still_accepted(monkeypatch):
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = LLMTreeRegressor(target=TARGET, features=FD, backend="openrouter").fit(X, np.arange(30, dtype=np.int64))
    assert est.predict(X).dtype == np.float64


def test_numpy_input_with_features(monkeypatch):
    # the features dict keys define the column names for NumPy input
    X, y = make_X_y()
    _patch_client(monkeypatch, make_float_tree_client(["a", "b"]))
    est = LLMTreeRegressor(target=TARGET, features=FD, backend="openrouter").fit(X.to_numpy(), y)
    assert est.feature_names_in_.tolist() == ["a", "b"]
    assert est.predict(X.to_numpy()).shape == (30,)


def test_clone_get_set_params():
    est = LLMTreeRegressor(
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
