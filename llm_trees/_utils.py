"""Shared fit/predict helpers: target/features validation, feature-name
resolution, prompt-file output, label mapping, complexity-constraint
validation, and the retry loop around tree generation.

The retry semantics: ``num_retry_llm`` is the total number of generation
attempts; a failure after the LLM call (post-processing, compilation,
parsing, complexity constraints, row-wise validation) triggers a new attempt,
and a ``ValueError`` is raised after the last attempt fails. Exceptions from
the LLM client itself (API/network/auth errors) propagate immediately without
retry.
"""

import numbers
import os

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import check_classification_targets

from ._tree import _check_node_vector, compile_tree_function, postprocess_prompting_result
from ._tree_ast import parse_tree_function


def check_target(target):
    """Validate the ``target`` parameter; return ``(name, description)``.

    ``target`` is a dict with exactly one entry mapping the target column
    name to its description (value encoding, units).
    """
    if not isinstance(target, dict) or len(target) != 1:
        raise ValueError(
            "target must be a dict with exactly one entry "
            "mapping the target column name to its description"
        )
    (name, description), = target.items()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("the target key (target column name) must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("the target description must be a non-empty string")
    return name, description


def check_target_name(target_name, y):
    """If y carries a column name, the target key must match it.

    A named Series (or a one-column DataFrame) names the target column, the
    same way a DataFrame names its feature columns; a nameless array/Series
    leaves the key descriptive.
    """
    y_name = None
    if isinstance(y, pd.DataFrame):
        if len(y.columns) == 1:
            y_name = y.columns[0]
    elif isinstance(y, pd.Series):
        y_name = y.name
    if y_name is not None and y_name != target_name:
        raise ValueError(f"target key {target_name!r} does not match the y column name {y_name!r}")


def check_features(features):
    """Validate the ``features`` parameter (dict: column name -> description)."""
    if not isinstance(features, dict) or not features:
        raise ValueError(
            "features must be a non-empty dict mapping each feature column name to its description"
        )
    for name, description in features.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"feature names must be non-empty strings, got {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"the feature description for {name!r} must be a non-empty string")
    return features


def resolve_feature_names(X, features):
    """Resolve the feature names to store as ``feature_names_in_``.

    DataFrame: the ``features`` keys must match ``X.columns`` exactly, the
    column order is canonical, and column names must be unique. NumPy
    array: the ``features`` keys define the names (one per column). Returns
    an object-dtype array.
    """
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            dupes = [str(c) for c in X.columns[X.columns.duplicated()]]
            raise ValueError(
                f"X has duplicate column names: {dupes}; column names must be "
                "unique, otherwise row-wise conversion would discard values"
            )
        missing = [c for c in X.columns if c not in features]
        unseen = [k for k in features if k not in X.columns]
        if missing or unseen:
            detail = []
            if missing:
                detail.append(f"missing: {missing}")
            if unseen:
                detail.append(f"unseen: {unseen}")
            raise ValueError(f"features keys must match X.columns ({'; '.join(detail)})")
        return np.asarray(list(X.columns), dtype=object)

    names = np.asarray(list(features.keys()), dtype=object)
    if len(names) != X.shape[1]:
        raise ValueError(f"features has {len(names)} entries, but X has {X.shape[1]} features")
    return names


def resolve_prompt_path(prompt_file):
    """Resolve the ``prompt_file`` parameter to a path, or None if disabled.

    ``True`` -> ``prompt.txt`` in the current working directory (the project
    root, next to the README and .env); a string -> that path; ``False``/None
    -> no file.
    """
    if prompt_file is True:
        return "prompt.txt"
    if prompt_file is False or prompt_file is None:
        return None
    return str(prompt_file)


def save_prompts(prompt_file, first_prompt, second_prompt):
    """Write both step prompts to ``prompt_file`` for inspection (no-op if disabled)."""
    path = resolve_prompt_path(prompt_file)
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== FIRST PROMPT (tree induction) =====\n")
        f.write(first_prompt)
        f.write("\n\n===== SECOND PROMPT (Python function generation) =====\n")
        f.write(second_prompt)
        f.write("\n")


def check_complexity_param(value, name):
    """Validate a complexity-constraint parameter (None or integer >= 1, not bool).

    Accepts any ``numbers.Integral`` (so numpy integer dtypes work).
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")


def check_classification_target(y) -> np.ndarray:
    """Normalize a supported classification target to a 1d array.

    Accepts 1d arrays/Series, ``(n, 1)`` arrays and one-column DataFrames
    (normalized to 1d); rejects multioutput ``(n, k>1)`` and
    higher-dimensional targets so they fail before any model request.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y should be a 1d classification target, got shape {y.shape}")
    check_classification_targets(y)
    return y


def check_regression_target(y) -> np.ndarray:
    """Validate a regression target: 1d, real-valued numeric (bool and
    complex not allowed), finite.

    Returns the target as a float64 array. Raises ``ValueError`` for
    multioutput, complex, non-numeric, and NaN/infinite values so they fail
    before any model request.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y should be a 1d array, got shape {y.shape}")
    # complex is a subtype of np.number and .astype(float64) would silently
    # discard the imaginary part, so it must be rejected before conversion
    if np.issubdtype(y.dtype, np.complexfloating):
        raise ValueError(f"y should be real-valued, got complex dtype {y.dtype}")
    if not (np.issubdtype(y.dtype, np.number) and not np.issubdtype(y.dtype, np.bool_)):
        raise ValueError(f"y should be numeric, got dtype {y.dtype}")
    y = y.astype(np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or infinity); regression targets must be finite")
    return y


def check_feature_names(X, feature_names_in_):
    """Predict-time check that X carries the feature names seen at fit time."""
    expected = list(np.asarray(feature_names_in_).tolist())
    if isinstance(X, pd.DataFrame):
        if list(X.columns) != expected:
            missing = [n for n in expected if n not in X.columns]
            unseen = [n for n in X.columns if n not in expected]
            detail = []
            if missing:
                detail.append(f"missing: {missing}")
            if unseen:
                detail.append(f"unseen: {unseen}")
            if not detail:
                detail.append("different order")
            raise ValueError(f"Feature names mismatch at predict time ({'; '.join(detail)})")
    elif X.shape[1] != len(expected):
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with {len(expected)} features"
        )


def map_label(value, classes):
    """Map a raw tree prediction to one of ``classes``.

    1. Exact match against ``classes`` is used as-is.
    2. Otherwise, an integral value is interpreted positionally
       (``0 <= value < len(classes)``), mirroring the "target value encoding"
       wording of the prompts.
    3. Anything else raises ``ValueError``.
    """
    if value in classes:
        return value
    if (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and 0 <= int(value) < len(classes)
    ):
        return classes[int(value)]
    raise ValueError(f"Invalid prediction label from LLM tree: {value!r}")


def is_numeric_prediction(value):
    """True for integral/floating predictions (bools excluded), finite only."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(value))


def check_complexity_constraints(pt, max_tree_depth, num_features):
    """Validate a parsed tree against the active complexity constraints.

    ``pt is None`` means the tree could not be parsed into the expected form,
    so its structure cannot be inspected: when a limit is configured the
    candidate is rejected (the attempt is retried); without configured limits
    it is accepted as-is.
    """
    if pt is None:
        if max_tree_depth is not None or num_features is not None:
            raise ValueError(
                "Tree structure could not be parsed; configured complexity "
                "constraints cannot be validated"
            )
        return
    if max_tree_depth is not None and pt.depth > max_tree_depth:
        raise ValueError(f"Tree violates max_tree_depth={max_tree_depth}")
    if num_features is not None and len(pt.features) > num_features:
        raise ValueError(f"Tree violates num_features={num_features}")


# ``_check_node_vector`` is imported from ``._tree`` (canonical definition;
# it validates the same ``np.asarray`` representation that embedding
# construction assigns into NumPy rows).


def _validate_predictions(fn, X, classes, numeric):
    """Run the tree function on every row; reject invalid predictions or node vectors.

    Each row must return a ``(prediction, nodes)`` tuple with a supported
    node vector of consistent width across rows.
    """
    num_nodes = None
    for i in range(len(X)):
        result = fn(X.iloc[i].to_dict())
        if not (isinstance(result, tuple) and len(result) == 2):
            raise ValueError(f"Tree function must return (prediction, nodes), got {type(result).__name__}")
        pred, nodes = result
        if classes is not None:
            map_label(pred, classes)
        elif numeric and not is_numeric_prediction(pred):
            raise ValueError(f"Invalid prediction label from LLM tree: {pred!r}")
        width = _check_node_vector(nodes)
        if num_nodes is None:
            num_nodes = width
        elif width != num_nodes:
            raise ValueError(f"Inconsistent node vector widths across rows: {num_nodes} vs {width}")


def generate_tree_with_retry(
    *,
    client,
    first_prompt,
    second_prompt,
    X,
    classes,
    max_tree_depth,
    num_features,
    num_retry_llm,
    numeric,
):
    """Generate, compile, and validate a tree function, retrying on failure.

    ``num_retry_llm`` is the total number of generation attempts. Each
    attempt: two-step LLM call, post-process, compile, parse, validate
    (complexity constraints, then row-wise predictions on ``X``). Returns
    ``(tree_fcn_str, ParsedTree | None, predict_fn)`` on the first valid
    attempt; raises ``ValueError`` after the last attempt fails.

    Only post-generation validation failures are retried. Exceptions raised
    by the LLM client itself (API/network/auth errors) propagate immediately.
    """
    last_error = None
    for _ in range(num_retry_llm):
        raw = client.two_step(first_prompt, second_prompt)
        try:
            code = postprocess_prompting_result(raw)
            fn = compile_tree_function(code)
            pt = parse_tree_function(code)
            check_complexity_constraints(pt, max_tree_depth, num_features)
            _validate_predictions(fn, X, classes=classes, numeric=numeric)
        except Exception as e:  # noqa: BLE001 - validation failures are retried
            last_error = e
            continue
        return code, pt, fn
    raise ValueError(f"Tree generation failed after {num_retry_llm} attempts: {last_error}")
