"""LLMTreeEmbeddingRegressor: tree-embedding induction + downstream model.

Induces ``num_trees`` decision trees via the two-step LLM workflow, embeds
samples as the concatenated per-tree node truth-value matrices (optionally
augmented with preprocessed raw features), and fits a downstream regressor
(a preset with its parameter grid, or a user-supplied estimator) on the
embedding.

The LLM is only called during ``fit``; ``transform`` and ``predict`` are pure
inference on the induced trees and fitted downstream model.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._embeddings import _TreeEmbeddingTransformer, fit_custom_downstream, fit_downstream
from ._llm import make_client
from ._prompts import build_first_prompt, build_second_prompt
from ._tree import compile_tree_function
from ._utils import (
    check_complexity_param,
    check_feature_names,
    check_features,
    check_regression_target,
    check_target,
    check_target_name,
    generate_tree_with_retry,
    resolve_feature_names,
    save_prompts,
)

_PRESET_ESTIMATORS = ("mlp", "hgbdt", "lr")


class LLMTreeEmbeddingRegressor(RegressorMixin, BaseEstimator):
    """Induce multiple LLM decision trees, embed samples, fit a probe regressor.

    Each sample is embedded by the truth values of the inner nodes of every
    induced tree (off-path nodes count as ``False``); optionally, preprocessed
    raw features (imputed, one-hot encoded, MinMax-scaled) are appended. A
    downstream regressor is then fitted on the embedding. No trees or models
    are written to disk, and no LLM calls happen after ``fit``.

    Parameters
    ----------
    target : dict
        One entry mapping the target column name to its description.
        Required: the LLM needs it to induce a meaningful tree. The
        key must match the name of ``y`` when ``y`` carries one
        (named Series or one-column DataFrame).

    features : dict
        One entry per feature mapping the column name to its description.
        Required: it must cover every feature of ``X``. Included in both
        step prompts, so the LLM sees each feature's name and description
        when inducing the trees.

    max_tree_depth : int or None, default=None
        Maximum tree depth requested in the prompt and enforced on each
        induced tree. None means no depth constraint.

    num_features : int or None, default=None
        Maximum number of distinct features requested in the prompt and
        enforced on each induced tree. None means no feature-count
        constraint. When either limit is set, an induced tree that cannot be
        parsed is rejected and generation is retried.

    num_trees : int, default=5
        Number of trees to induce (one LLM two-step call per tree attempt).

    append_raw_features : bool, default=True
        Whether to append preprocessed raw features to the node embeddings.

    estimator : str, default="mlp"
        Downstream regressor preset: "mlp", "hgbdt", or "lr". Ignored when
        ``downstream_estimator`` is given.

    downstream_estimator : regressor or None, default=None
        A user-supplied downstream estimator fitted directly (no grid
        search). When set, ``estimator`` is ignored.

    categorical_features : list[str] or None, default=None
        Feature names to treat as nominal in the raw-feature preprocessor.
        None means a dtype-based split (numeric dtypes numerical, the rest
        nominal).

    prompt_file : str or bool, default=True
        Where to write the generated step prompts for inspection: True writes
        ``prompt.txt`` to the current working directory (your project root);
        a string writes to that path (parent directories are created);
        False disables the file.

    backend : str, default="openrouter"
        LLM backend: "openrouter" (the only supported backend; requires the
        ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` environment variables).

    temperature : float, default=1.0
        Sampling temperature forwarded to the LLM backend.

    seed : int or None, default=None
        Seed forwarded to the downstream model.

    num_examples : int, default=1
        Number of example ``predict`` functions included in the prompts
        (0 disables examples).

    use_role_prompt : bool, default=False
        Whether to prefix the first prompt with the role prompt.

    num_retry_llm : int, default=10
        Total number of generation attempts per tree; a ``ValueError`` is
        raised if every attempt for a tree fails.

    Attributes
    ----------
    trees_ : list of (str, callable, ParsedTree or None)
        The induced trees: compiled source, predict function, and parsed
        structure (None when unparsable).

    feature_names_in_ : ndarray of shape (n_features,), dtype=object
        Feature names seen at fit time.

    support_ : ndarray of shape (sparsity,), dtype=object
        Sorted union of the feature names referenced by the induced trees.

    sparsity_ : int
        Size of ``support_``.

    elapsed_time_ : float
        Wall-clock fit time in seconds (includes LLM calls).

    final_estimator_ : Pipeline
        The fitted ``Pipeline([("transformer", ...), ("model", ...)])``;
        ``predict`` delegates to it (the transformer is applied inside).

    Notes
    -----
    ``transform`` returns the embedding matrix (n_samples x sum of per-tree
    node counts, plus preprocessed raw columns when enabled). After
    ``fit`` returns, ``transform`` and ``predict`` never call the LLM.
    """

    def __init__(
        self,
        target: dict,
        features: dict,
        max_tree_depth: int | None = None,
        num_features: int | None = None,
        num_trees: int = 5,
        append_raw_features: bool = True,
        estimator: str = "mlp",
        downstream_estimator=None,
        categorical_features: list[str] | None = None,
        prompt_file: str | bool = True,
        backend: str = "openrouter",
        temperature: float = 1.0,
        seed: int | None = None,
        num_examples: int = 1,
        use_role_prompt: bool = False,
        num_retry_llm: int = 10,
    ):
        self.target = target
        self.features = features
        self.max_tree_depth = max_tree_depth
        self.num_features = num_features
        self.num_trees = num_trees
        self.append_raw_features = append_raw_features
        self.estimator = estimator
        self.downstream_estimator = downstream_estimator
        self.categorical_features = categorical_features
        self.prompt_file = prompt_file
        self.backend = backend
        self.temperature = temperature
        self.seed = seed
        self.num_examples = num_examples
        self.use_role_prompt = use_role_prompt
        self.num_retry_llm = num_retry_llm

    def __getstate__(self):
        # compiled tree functions are not picklable; the stored source code
        # is the single source of truth (``final_estimator_``'s transformer
        # restores its own copies via its state protocol)
        state = self.__dict__.copy()
        if state.get("trees_") is not None:
            state["trees_"] = [(code, None, pt) for code, _fn, pt in state["trees_"]]
        return state

    def __setstate__(self, state):
        trees = state.get("trees_")
        if trees is not None:
            state["trees_"] = [
                (code, compile_tree_function(code), pt) for code, _fn, pt in trees
            ]
        self.__dict__.update(state)

    def fit(self, X, y):
        """Induce the trees and fit the downstream regressor on the embedding.

        Learned state is committed to the estimator only after a successful
        fit; a failed fit leaves the previous state (fitted or unfitted)
        untouched, so inference never sees a mixed old/new state.
        """
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples, but y has {len(y)} samples")
        start = time.monotonic()

        target_name, _target_desc = check_target(self.target)
        check_target_name(target_name, y)
        check_complexity_param(self.max_tree_depth, "max_tree_depth")
        check_complexity_param(self.num_features, "num_features")
        if isinstance(self.num_trees, bool) or not isinstance(self.num_trees, int) or self.num_trees < 1:
            raise ValueError(f"num_trees must be a positive integer, got {self.num_trees!r}")
        if self.downstream_estimator is None and self.estimator not in _PRESET_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator preset: {self.estimator!r}; "
                f"use one of {_PRESET_ESTIMATORS} or pass downstream_estimator"
            )
        check_features(self.features)

        feature_names_in = resolve_feature_names(X, self.features)
        y = check_regression_target(y)

        X = self._as_frame(X, feature_names_in)
        features = {name: self.features[name] for name in feature_names_in}
        first_prompt = build_first_prompt(
            target=self.target,
            features=features,
            max_tree_depth=self.max_tree_depth,
            num_features=self.num_features,
            num_examples=self.num_examples,
            use_role_prompt=self.use_role_prompt,
        )
        second_prompt = build_second_prompt(
            target=self.target,
            features=features,
            num_examples=self.num_examples,
            max_tree_depth=self.max_tree_depth,
        )
        save_prompts(self.prompt_file, first_prompt, second_prompt)

        trees = []
        for tree_idx in range(self.num_trees):
            # induction of tree i uses seed i (one two-step dialogue per tree)
            client = make_client(self.backend, self.temperature, tree_idx)
            code, pt, fn = generate_tree_with_retry(
                client=client,
                first_prompt=first_prompt,
                second_prompt=second_prompt,
                X=X,
                classes=None,
                max_tree_depth=self.max_tree_depth,
                num_features=self.num_features,
                num_retry_llm=self.num_retry_llm,
                numeric=True,
            )
            trees.append((code, fn, pt))

        transformer = _TreeEmbeddingTransformer(
            trees, self.append_raw_features, self.categorical_features, self.seed
        ).fit(X)
        if self.downstream_estimator is None:
            final_estimator = fit_downstream(
                "regressor", self.estimator, X, y, transformer, self.seed
            )
        else:
            final_estimator = fit_custom_downstream(X, y, transformer, self.downstream_estimator)

        all_features = sorted({f for _c, _fn, pt in trees if pt is not None for f in pt.features})
        support = np.asarray(all_features, dtype=object)

        # commit learned state only after successful generation and validation
        self.feature_names_in_ = feature_names_in
        self.trees_ = trees
        self.final_estimator_ = final_estimator
        self.support_ = support
        self.sparsity_ = len(support)
        self.elapsed_time_ = time.monotonic() - start
        return self

    def _as_frame(self, X, feature_names_in) -> pd.DataFrame:
        """Validate feature names and return X as a DataFrame (row dicts)."""
        check_feature_names(X, feature_names_in)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names_in)
        return X

    def transform(self, X):
        """Embedding matrix for ``X`` (node truth values [+ raw features])."""
        check_is_fitted(self)
        X = self._as_frame(X, self.feature_names_in_)
        return self.final_estimator_.named_steps["transformer"].transform(X)

    def predict(self, X):
        """Predict float values for ``X`` via the fitted downstream model."""
        check_is_fitted(self)
        X = self._as_frame(X, self.feature_names_in_)
        return self.final_estimator_.predict(X)
