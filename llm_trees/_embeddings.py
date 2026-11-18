"""In-memory tree embedding transformer and downstream model presets.

The transformer embeds each sample with the truth values of the inner nodes
of the induced trees (off-path nodes count as ``False``) and optionally
appends preprocessed raw features (imputed, one-hot encoded, MinMax-scaled).
Downstream presets are fixed classifier/regressor pipelines fitted in-memory
with their parameter grids; no trees or models are written to disk.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ._tree import _check_node_vector, compile_tree_function


def build_preprocessor(X, categorical_features=None):
    """Build the raw-feature preprocessor (nominal + numerical pipelines).

    ``categorical_features=None`` -> dtype-based split (numeric dtypes are
    numerical, everything else nominal). Column groups that end up empty are
    omitted.
    """
    columns = list(X.columns)
    if categorical_features is None:
        categorical_features = [c for c in columns if not pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = list(categorical_features)
    numerical_features = [c for c in columns if c not in categorical_features]

    transformers = []
    if categorical_features:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
                        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if numerical_features:
        transformers.append(
            (
                "numerical",
                Pipeline(
                    [
                        ("numerical_imputer", KNNImputer(n_neighbors=10, keep_empty_features=True)),
                        ("numerical_scaler", MinMaxScaler()),
                    ]
                ),
                numerical_features,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0)


def check_scaling(X_transformed):
    """Range guard: preprocessed features must stay within [-0.1, 1.1]."""
    x_min = float(np.min(X_transformed))
    x_max = float(np.max(X_transformed))
    if x_min < -0.1 or x_max > 1.1:
        raise ValueError(f"Invalid range: [{x_min}, {x_max}]")


class _TreeEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Embed X as the per-tree node-truth-value matrices (Borisov et al. 2023).

    Each tree contributes its ``nodes`` matrix (off-path ``None`` entries are
    stored as ``False``); optionally, preprocessed raw features are appended.
    """

    def __init__(self, trees, append_raw_features, categorical_features=None, random_state=None):
        self.trees = trees
        self.append_raw_features = append_raw_features
        self.categorical_features = categorical_features
        self.random_state = random_state

    def _restore_fns(self, state):
        for key in ("trees", "trees_"):
            trees = state.get(key)
            if trees is not None:
                state[key] = [(code, compile_tree_function(code), pt) for code, _fn, pt in trees]

    def __getstate__(self):
        # compiled tree functions are not picklable; the stored source code
        # is the single source of truth
        state = self.__dict__.copy()
        for key in ("trees", "trees_"):
            if state.get(key) is not None:
                state[key] = [(code, None, pt) for code, _fn, pt in state[key]]
        return state

    def __setstate__(self, state):
        self._restore_fns(state)
        self.__dict__.update(state)

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.trees_ = list(self.trees)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        # Fitted per-tree node vector widths, re-checked at transform time.
        # Unknown when fit on an empty X (documented limitation).
        if len(X) == 0:
            self._node_widths_ = [None] * len(self.trees_)
        else:
            self._node_widths_ = [
                _check_node_vector(fn(X.iloc[0].to_dict())[1]) for _code, fn, _pt in self.trees_
            ]
        if self.append_raw_features:
            self.preprocessor_ = build_preprocessor(X, self.categorical_features)
            self.preprocessor_.fit(X)
            check_scaling(self.preprocessor_.transform(X))
        return self

    def _as_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X

    def transform(self, X):
        check_is_fitted(self)
        X = self._as_frame(X)
        parts = [
            self._tree_nodes_matrix(fn, X, fitted_width)
            for (_code, fn, _pt), fitted_width in zip(self.trees_, self._node_widths_)
        ]
        if self.append_raw_features:
            parts.append(self.preprocessor_.transform(X))
        return np.hstack(parts)

    def _tree_nodes_matrix(self, fn, X, fitted_width):
        """Per-row node truth values; every row's vector must have the fitted width.

        The first row is evaluated unconditionally: an empty ``X`` raises
        ``IndexError``, and a fit on an empty X (unknown width) takes it from
        the first inference row. Off-path ``None`` entries become ``False``
        on assignment.
        """
        _y, row = fn(X.iloc[0].to_dict())
        width = _check_node_vector(row)
        if fitted_width is None:
            fitted_width = width
        nodes = np.zeros((len(X), fitted_width), dtype=bool)
        for i in range(len(X)):
            if i > 0:
                _y, row = fn(X.iloc[i].to_dict())
                width = _check_node_vector(row)
            if width != fitted_width:
                raise ValueError(
                    f"Node width changed at inference time: fitted {fitted_width}, "
                    f"got {width} for row {i}"
                )
            nodes[i, :] = row
        return nodes.astype(float)


def downstream_preset(kind, preset, random_state):
    """Return ``(model, param_grid, scoring)`` for a downstream preset.

    ``param_grid`` is ``None`` when the preset is fitted directly (no grid
    search).
    """
    presets = {
        ("classifier", "mlp"): (
            lambda rs: MLPClassifier(solver="lbfgs", max_iter=10000, random_state=rs),
            {
                "model__hidden_layer_sizes": [10, 25, 50, 75, 100],
                "model__alpha": [1e-4, 1e-3, 0.01, 0.1, 1],
            },
            "f1_macro",
        ),
        ("classifier", "hgbdt"): (
            lambda rs: HistGradientBoostingClassifier(random_state=rs),
            {"model__min_samples_leaf": [5, 10, 15, 20, 25]},
            "f1_macro",
        ),
        ("classifier", "lr"): (
            lambda rs: LogisticRegression(max_iter=10000, random_state=rs),
            {"model__C": [1, 10, 100, 1000, 10000]},
            "f1_macro",
        ),
        ("regressor", "mlp"): (
            lambda rs: MLPRegressor(solver="lbfgs", max_iter=10000, random_state=rs),
            {
                "model__hidden_layer_sizes": [10, 25, 50, 75, 100],
                "model__alpha": [1e-4, 1e-3, 0.01, 0.1, 1],
            },
            "r2",
        ),
        ("regressor", "hgbdt"): (
            lambda rs: HistGradientBoostingRegressor(random_state=rs),
            {"model__min_samples_leaf": [5, 10, 15, 20, 25]},
            "r2",
        ),
        ("regressor", "lr"): (
            lambda rs: LinearRegression(),
            None,
            "r2",
        ),
    }
    key = (kind, preset)
    if key not in presets:
        raise ValueError(f"Unknown downstream preset: {preset!r} for {kind!r}")
    model_factory, param_grid, scoring = presets[key]
    return model_factory(random_state), param_grid, scoring


def fit_downstream(kind, preset, X, y, transformer, random_state):
    """Fit ``Pipeline([("transformer", ...), ("model", ...)])`` with the preset.

    Returns the fitted pipeline (``GridSearchCV.best_estimator_`` for
    grid-searched presets, the fitted pipeline for direct-fit presets).
    """
    model, param_grid, scoring = downstream_preset(kind, preset, random_state)
    pipeline = Pipeline([("transformer", transformer), ("model", model)])
    if param_grid is None:
        pipeline.fit(X, y)
        return pipeline
    search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, cv=3, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


def fit_custom_downstream(X, y, transformer, downstream_estimator):
    """Fit the pipeline with a user-supplied downstream estimator (no grid)."""
    pipeline = Pipeline([("transformer", transformer), ("model", clone(downstream_estimator))])
    pipeline.fit(X, y)
    return pipeline
