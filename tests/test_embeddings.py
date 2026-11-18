"""Tests for llm_trees._embeddings: embedding transformer and downstream presets.

Uses scripted deterministic tree functions (no LLM calls).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from conftest import make_scripted_tree
from llm_trees._embeddings import (
    _TreeEmbeddingTransformer,
    build_preprocessor,
    check_scaling,
    downstream_preset,
    fit_custom_downstream,
    fit_downstream,
)
from llm_trees._tree import predict_rows

TWO_NODE_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    '        prediction = "yes"\n'
    "    else:\n"
    "        if nodes[1]:\n"
    '            prediction = "no"\n'
    "        else:\n"
    '            prediction = "maybe"\n'
    "    return prediction, nodes\n"
)

THREE_NODE_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 3 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["a"] >= 2.0\n'
    '    nodes[2] = X["b"] <= 5.0\n'
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
)

LAZY_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    "    if nodes[0]:\n"
    '        prediction = "left"\n'
    "    else:\n"
    '        nodes[1] = X["b"] <= 2.0\n'
    "        if nodes[1]:\n"
    '            prediction = "br"\n'
    "        else:\n"
    '            prediction = "bl"\n'
    "    return prediction, nodes\n"
)


def basic_trees():
    return [make_scripted_tree(TWO_NODE_CODE), make_scripted_tree(THREE_NODE_CODE)]


def basic_X():
    return pd.DataFrame({"a": [0.5, 1.5, 2.5, 3.0], "b": [1.0, 2.0, 6.0, 7.0]})


def test_transformer_shape_nodes_only():
    t = _TreeEmbeddingTransformer(basic_trees(), append_raw_features=False).fit(basic_X())
    emb = t.transform(basic_X())
    assert emb.shape == (4, 5)  # 2 nodes + 3 nodes
    assert emb.dtype == np.float64
    assert set(np.unique(emb)) <= {0.0, 1.0}


def test_transformer_shape_with_raw_features():
    X = pd.DataFrame(
        {"a": [0.5, 1.5, 2.5, 3.0], "b": [1.0, 2.0, 6.0, 7.0], "c": [1.0, 2.0, 3.0, 4.0], "d": [4.0, 3.0, 2.0, 1.0]}
    )
    t = _TreeEmbeddingTransformer(basic_trees(), append_raw_features=True).fit(X)
    emb = t.transform(X)
    assert emb.shape == (4, 5 + 4)  # nodes + 4 scaled numeric
    # MinMax-scaled within the check_scaling guard (float rounding)
    assert (emb[:, 5:] >= -0.1).all() and (emb[:, 5:] <= 1.1).all()

    # one categorical column -> one-hot block (nominal group is declared first)
    X2 = X.assign(cat=["x", "y", "x", "y"])
    t2 = _TreeEmbeddingTransformer(basic_trees(), append_raw_features=True).fit(X2)
    emb2 = t2.transform(X2)
    assert emb2.shape == (4, 5 + 2 + 4)  # nodes + 2 one-hot + 4 scaled
    assert set(np.unique(emb2[:, 5:7])) <= {0.0, 1.0}
    assert (emb2[:, 7:] >= -0.1).all() and (emb2[:, 7:] <= 1.1).all()


def test_embedding_nodes_none_becomes_false():
    X = pd.DataFrame({"a": [0.5, 1.5], "b": [9.0, 1.0]})
    t = _TreeEmbeddingTransformer([make_scripted_tree(LAZY_CODE)], append_raw_features=False).fit(X)
    emb = t.transform(X)
    assert emb.shape == (2, 2)
    assert not np.isnan(emb).any()
    assert set(np.unique(emb)) <= {0.0, 1.0}
    # off-path None node on row 0 -> stored as False
    assert emb[0, 0] == 1.0 and emb[0, 1] == 0.0


def test_preprocessor_auto_dtype_split():
    X = pd.DataFrame({"n1": [1.0, 2.0, 3.0, 4.0], "n2": [4, 3, 2, 1], "cat": ["b", "a", "b", "a"]})
    prep = build_preprocessor(X)
    out = prep.fit_transform(X)
    assert out.shape == (4, 4)  # 2 one-hot (cat) + 2 scaled numeric
    assert out.dtype == np.float64
    assert set(np.unique(out[:, :2])) <= {0.0, 1.0}  # nominal (declared first)
    assert (out[:, 2:] >= -0.1).all() and (out[:, 2:] <= 1.1).all()  # check_scaling guard

    # explicit categorical_features list takes precedence
    prep2 = build_preprocessor(X, categorical_features=["n1", "cat"])
    out2 = prep2.fit_transform(X)
    assert out2.shape == (4, 6 + 1)  # one-hot(n1: 4 + cat: 2) + scaled(n2: 1)


def test_transform_rejects_changed_node_width():
    # the function returns a 1-node vector for small rows and a 2-node vector
    # for large rows: fit rows define the width, inference must check it
    code = (
        "def predict(X: dict):\n"
        '    if X["a"] < 10.0:\n'
        '        return 0, [X["a"] <= 1.0]\n'
        "    else:\n"
        '        return 0, [X["a"] <= 1.0, X["a"] >= 0.0]\n'
    )
    trees = [make_scripted_tree(code)]
    X_fit = pd.DataFrame({"a": [1.0, 2.0]})
    t = _TreeEmbeddingTransformer(trees, False).fit(X_fit)
    assert t.transform(X_fit).shape == (2, 1)
    with pytest.raises(ValueError, match="width"):
        t.transform(pd.DataFrame({"a": [50.0]}))


def test_check_scaling_raises_out_of_range():
    check_scaling(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="Invalid range"):
        check_scaling(np.array([[0.0, 2.0]]))
    with pytest.raises(ValueError, match="Invalid range"):
        check_scaling(np.array([[-0.5, 0.0]]))


def test_classifier_presets_construct_correct_model_and_grid():
    expected = {
        "mlp": (MLPClassifier, {"model__hidden_layer_sizes", "model__alpha"}),
        "hgbdt": (HistGradientBoostingClassifier, {"model__min_samples_leaf"}),
        "lr": (LogisticRegression, {"model__C"}),
    }
    for preset, (model_cls, grid_keys) in expected.items():
        model, grid, scoring = downstream_preset("classifier", preset, 42)
        assert isinstance(model, model_cls)
        assert set(grid) == grid_keys
        assert scoring == "f1_macro"

    # one end-to-end fit (lr is fast)
    X = basic_X()
    y = pd.Series(["a", "b", "a", "b"])
    X_big = pd.concat([X] * 8, ignore_index=True)  # 32 rows for cv=3
    y_big = pd.Series(["a", "b"] * 16)
    est = fit_downstream("classifier", "lr", X_big, y_big, _TreeEmbeddingTransformer(basic_trees(), False).fit(X_big), 42)
    assert isinstance(est, Pipeline)
    assert isinstance(est.named_steps["model"], LogisticRegression)
    assert est.predict(X_big).shape == (32,)


def test_regressor_presets_construct_correct_model():
    expected = {
        "mlp": (MLPRegressor, {"model__hidden_layer_sizes", "model__alpha"}),
        "hgbdt": (HistGradientBoostingRegressor, {"model__min_samples_leaf"}),
        "lr": (LinearRegression, None),
    }
    for preset, (model_cls, grid_keys) in expected.items():
        model, got_grid, scoring = downstream_preset("regressor", preset, 42)
        assert isinstance(model, model_cls)
        if grid_keys is None:
            assert got_grid is None
        else:
            assert set(got_grid) == grid_keys
        assert scoring == "r2"
    with pytest.raises(ValueError, match="Unknown downstream preset"):
        downstream_preset("classifier", "xgb", 42)


def test_user_downstream_estimator_bypasses_grid():
    class RecordingClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.X_fit_ = np.asarray(X)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X = basic_X()
    trees = basic_trees()
    y = pd.Series(["a", "b", "a", "b"])
    est = fit_custom_downstream(X, y, _TreeEmbeddingTransformer(trees, False).fit(X), RecordingClassifier())
    assert isinstance(est, Pipeline)
    assert isinstance(est.named_steps["model"], RecordingClassifier)
    # captured input equals the manual hstack of the node matrices
    manual = np.hstack([predict_rows(fn, X)[1].astype(float) for _c, fn, _p in trees])
    np.testing.assert_array_equal(est.named_steps["model"].X_fit_, manual)
    assert est.predict(X).shape == (4,)
