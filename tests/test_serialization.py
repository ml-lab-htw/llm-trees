"""Serialization (pickle/joblib) roundtrip tests for the four fitted estimators.

Dynamically generated tree functions are reconstructed from their stored
source on load: no retraining, no LLM calls. Roundtrips must leave
predictions, embeddings, and probabilities unchanged.
"""

import io
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from joblib.numpy_pickle import NumpyPickler, NumpyUnpickler
from sklearn.exceptions import NotFittedError

from conftest import DIALOGUE_SEPARATOR, ScriptedClient
from llm_trees import (
    LLMTreeClassifier,
    LLMTreeEmbeddingClassifier,
    LLMTreeEmbeddingRegressor,
    LLMTreeRegressor,
)

TARGET = {"y": "a continuous target"}

# features dict matching the X columns fitted in this file
# (the description text is irrelevant to these tests; only coverage is checked)
FD = {"a": "feature a", "b": "feature b"}


def _raw(fenced_code: str) -> str:
    return "A decision tree.\n\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + fenced_code


CLS_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    "    nodes = 1 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "normal"\n'
    "    else:\n"
    '        prediction = "abnormal"\n'
    "    return prediction, nodes\n"
    "```\n"
)

REG_CODE = (
    "```python\n"
    "def predict(X: dict):\n"
    '    return float(X["a"] * 2 + X["b"]), [X["a"] <= 1.0]\n'
    "```\n"
)


def _cls_X_y():
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


def _roundtrip(est, backend):
    if backend == "pickle":
        return pickle.loads(pickle.dumps(est))
    buf = io.BytesIO()
    NumpyPickler(buf).dump(est)
    buf.seek(0)
    return NumpyUnpickler("", buf, True).load()


BACKENDS = ["pickle", "joblib"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_induction_classifier_roundtrip(monkeypatch, backend):
    X, y = _cls_X_y()
    client = ScriptedClient(_raw(CLS_CODE))
    monkeypatch.setattr(
        "llm_trees._llm_tree_classifier.make_client", lambda m, t, s: client
    )
    est = LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
    pred = est.predict(X)
    proba = est.predict_proba(X)

    clone = _roundtrip(est, backend)
    np.testing.assert_array_equal(clone.predict(X), pred)
    np.testing.assert_array_equal(clone.predict_proba(X), proba)
    np.testing.assert_array_equal(clone.classes_, est.classes_)
    assert clone.tree_ is not None  # parsed structure survives the roundtrip


@pytest.mark.parametrize("backend", BACKENDS)
def test_induction_regressor_roundtrip(monkeypatch, backend):
    X, _ = _cls_X_y()
    y = (2.0 * X["a"] + X["b"]).to_numpy()
    client = ScriptedClient(_raw(REG_CODE))
    monkeypatch.setattr(
        "llm_trees._llm_tree_regressor.make_client", lambda m, t, s: client
    )
    est = LLMTreeRegressor(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
    pred = est.predict(X)
    assert pred.dtype == np.float64

    clone = _roundtrip(est, backend)
    np.testing.assert_array_equal(clone.predict(X), pred)
    assert clone.predict(X).dtype == np.float64


@pytest.mark.parametrize("backend", BACKENDS)
def test_embedding_classifier_roundtrip(monkeypatch, backend):
    X, y = _cls_X_y()
    client = ScriptedClient(_raw(CLS_CODE))
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_classifier.make_client", lambda m, t, s: client
    )
    est = LLMTreeEmbeddingClassifier(
        target=TARGET, features=FD, prompt_file=False, backend="openrouter", num_trees=2, estimator="lr",
        append_raw_features=False,
    ).fit(X, y)
    pred = est.predict(X)
    emb = est.transform(X)

    clone = _roundtrip(est, backend)
    np.testing.assert_array_equal(clone.predict(X), pred)
    np.testing.assert_array_equal(clone.transform(X), emb)
    np.testing.assert_array_equal(clone.classes_, est.classes_)


@pytest.mark.parametrize("backend", BACKENDS)
def test_embedding_regressor_roundtrip(monkeypatch, backend):
    X, _ = _cls_X_y()
    y = (2.0 * X["a"] + X["b"]).to_numpy()
    client = ScriptedClient(_raw(REG_CODE))
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_regressor.make_client", lambda m, t, s: client
    )
    est = LLMTreeEmbeddingRegressor(
        target=TARGET, features=FD, prompt_file=False, backend="openrouter", num_trees=2, estimator="lr",
        append_raw_features=False,
    ).fit(X, y)
    pred = est.predict(X)
    emb = est.transform(X)

    clone = _roundtrip(est, backend)
    np.testing.assert_array_equal(clone.predict(X), pred)
    np.testing.assert_array_equal(clone.transform(X), emb)
    assert clone.predict(X).dtype == np.float64


@pytest.mark.parametrize("backend", BACKENDS)
def test_unfitted_roundtrip_stays_unfitted(backend):
    X, _ = _cls_X_y()
    clone = _roundtrip(LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False), backend)
    with pytest.raises(NotFittedError):
        clone.predict(X)
    clone = _roundtrip(LLMTreeEmbeddingClassifier(target=TARGET, features=FD, prompt_file=False), backend)
    with pytest.raises(NotFittedError):
        clone.predict(X)


def test_file_based_joblib_dump_load_roundtrip(monkeypatch, tmp_path):
    X, y = _cls_X_y()
    yr = (2.0 * X["a"] + X["b"]).to_numpy()
    monkeypatch.setattr(
        "llm_trees._llm_tree_classifier.make_client",
        lambda m, t, s: ScriptedClient(_raw(CLS_CODE)),
    )
    c = LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
    monkeypatch.setattr(
        "llm_trees._llm_tree_regressor.make_client",
        lambda m, t, s: ScriptedClient(_raw(REG_CODE)),
    )
    r = LLMTreeRegressor(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, yr)
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_classifier.make_client",
        lambda m, t, s: ScriptedClient(_raw(CLS_CODE)),
    )
    ec = LLMTreeEmbeddingClassifier(
        target=TARGET, features=FD, prompt_file=False, backend="openrouter", num_trees=1, estimator="lr",
        append_raw_features=False,
    ).fit(X, y)
    monkeypatch.setattr(
        "llm_trees._llm_tree_embedding_regressor.make_client",
        lambda m, t, s: ScriptedClient(_raw(REG_CODE)),
    )
    er = LLMTreeEmbeddingRegressor(
        target=TARGET, features=FD, prompt_file=False, backend="openrouter", num_trees=1, estimator="lr",
        append_raw_features=False,
    ).fit(X, yr)

    checks = [
        (c, lambda e: (e.predict(X), e.predict_proba(X))),
        (r, lambda e: (e.predict(X),)),
        (ec, lambda e: (e.predict(X), e.transform(X))),  # embedding pipeline
        (er, lambda e: (e.predict(X), e.transform(X))),  # embedding pipeline
    ]
    for i, (est, get) in enumerate(checks):
        before = get(est)
        path = tmp_path / f"est_{i}.joblib"
        joblib.dump(est, path)
        loaded = joblib.load(path)
        for b, a in zip(before, get(loaded)):
            np.testing.assert_array_equal(a, b)


def test_no_llm_client_stored_on_fitted_estimator(monkeypatch):
    # the fit's LLM client must not be part of the fitted state: unpickling
    # can therefore never trigger a model request
    X, y = _cls_X_y()
    client = ScriptedClient(_raw(CLS_CODE))
    monkeypatch.setattr(
        "llm_trees._llm_tree_classifier.make_client", lambda m, t, s: client
    )
    est = LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
    state = est.__getstate__()
    assert "client" not in state and not any(
        isinstance(v, ScriptedClient) for v in state.values()
    )
    # reconstructing the function from stored source: no client, no network
    est.predict(X)
