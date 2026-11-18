# LLM Trees: Decision Trees with Language Models

Official repo:
["Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models](https://dl.acm.org/doi/10.1145/3711896.3736818).

Large language models (LLMs) provide powerful means to leverage prior knowledge for predictive modeling when data is limited.
In this work, we demonstrate how LLMs can use their compressed world knowledge to generate intrinsically interpretable machine
learning models, i.e., decision trees, without any training data. We find that these zero-shot decision trees can even surpass
data-driven trees on some small-sized tabular datasets and that embeddings derived from these trees perform better than data-driven
tree-based embeddings on average. Our knowledge-driven decision tree induction and embedding approaches therefore serve as strong
new baselines for data-driven machine learning methods in the low-data regime. Furthermore, they offer ways to harness the rich
world knowledge within LLMs for tabular machine learning tasks.

`llm-trees` exposes this zero-shot LLM decision tree induction and embedding as
four scikit-learn compatible estimators:

- `LLMTreeClassifier` — induce one decision tree classifier.
- `LLMTreeRegressor` — induce one decision tree regressor.
- `LLMTreeEmbeddingClassifier` — induce several trees, embed samples as node truth values, fit a downstream classifier.
- `LLMTreeEmbeddingRegressor` — induce several trees, embed samples as node truth values, fit a downstream regressor.

## Installation

```sh
pip install llm-trees
```

The LLM-generated prediction code is executed as Python in your process; run `llm-trees` in a sandbox if you induce trees
from data or descriptions you do not fully trust.

## Configuration

LLM calls go through [OpenRouter](https://openrouter.ai) via its
OpenAI-compatible API. Configure it with the environment variables
`OPENROUTER_API_KEY` and `OPENROUTER_MODEL` — either in your environment or in
a `.env` file in your project root (loaded via `python-dotenv`):

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=model/name
```

To use another OpenAI-compatible server (vLLM, LiteLLM, Ollama), also set
`OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`); `OPENROUTER_API_KEY` is still sent as the bearer token.

The variables are checked lazily: importing `llm_trees` and constructing
estimators never requires credentials; they are only needed when
`backend="openrouter"` actually calls the API.

## Estimators

All estimators accept `X` as a pandas `DataFrame` or a NumPy array (the
`features` dict provides the column names).

```python
LLMTreeClassifier(
    target, features,
    max_tree_depth=None, num_features=None, prompt_file=True,
    backend="openrouter", temperature=0.0, seed=None, num_examples=1,
    use_role_prompt=False, num_retry_llm=10,
)

LLMTreeRegressor(
    target, features,
    max_tree_depth=None, num_features=None, prompt_file=True,
    backend="openrouter", temperature=0.0, seed=None, num_examples=1,
    use_role_prompt=False, num_retry_llm=10,
)

LLMTreeEmbeddingClassifier(
    target, features,
    max_tree_depth=None, num_features=None, num_trees=5,
    append_raw_features=True, estimator="mlp", downstream_estimator=None,
    categorical_features=None, prompt_file=True, backend="openrouter",
    temperature=1.0, seed=None, num_examples=1, use_role_prompt=False,
    num_retry_llm=10,
)

LLMTreeEmbeddingRegressor(
    target, features,
    max_tree_depth=None, num_features=None, num_trees=5,
    append_raw_features=True, estimator="mlp", downstream_estimator=None,
    categorical_features=None, prompt_file=True, backend="openrouter",
    temperature=1.0, seed=None, num_examples=1, use_role_prompt=False,
    num_retry_llm=10,
)
```

Key parameters:

- `target` (required): a dict with exactly one entry mapping the target
  column name to its description, including the value encoding for
  classification (e.g., `{"churn": "customer churn (no, yes)"}` or
  `{"churn": "customer churn (0 = no, 1 = yes)"}`) or the measurement
  units for regression (e.g., `{"salary": "annual salary (EUR)"}` or
  `{"sf36_pf": "SF-36 physical functioning subscore (unitless)"}`).
  The key is enforced to match the name of `y` when `y` carries one
  (named Series or one-column DataFrame); with a nameless Series or
  array the key is descriptive.
- `features` (required): a dict with one entry per feature, mapping each
  column name to its description, including the value encoding for
  categorical features and measurement units for numerical features
  (e.g., `{"sex": "biological sex (male, female)",
  "ad_grade": "admission grade (1 = very good, 2 = good, 3 = satisfactory,
  4 = sufficient, 5 = deficient, 6 = insufficient)",
  "age": "age (years)",
  "ab_grade": "Abitur average grade (unitless)"}`); it must cover every
  feature of `X` (DataFrame: keys must equal `X.columns`; NumPy array:
  one key per column).
- `backend`: `"openrouter"` is the only supported backend; LLM calls go
  through OpenRouter and require the `OPENROUTER_API_KEY` and
  `OPENROUTER_MODEL` environment variables.
- `temperature`: sampling temperature forwarded to the LLM backend;
   a `temperature` of `0` gives more deterministic responses, but the
   degree of determinism depends on the LLM
   [THINK AI FIRST, 2026](https://think-ai-first.com/blog/llm-temperature-zero-silent-divergence).

### Induction vs. embedding

- **Induction estimators** (`LLMTreeClassifier`, `LLMTreeRegressor`) induce a
  single decision tree during `fit` and predict with that tree directly:
  `predict` executes the generated `predict(X)` function row-wise and returns
  the original target labels (classification) or float values (regression).
  `LLMTreeClassifier` additionally provides `predict_proba`, the fraction of
  fit samples of each class in the leaf a sample lands in (empty leaves fall
  back to the class prior). `predict_proba` (and the structural attributes
  `tree_`, `support_`, `sparsity_`) requires a structurally parsable tree:
  when no complexity constraint is set, an executable but unparsable `predict(X)`
  function may be accepted (see *Two-step workflow* below), in which case
  the structural attributes are unavailable and `predict_proba` raises
  `RuntimeError`.
- **Embedding estimators** (`LLMTreeEmbeddingClassifier`,
  `LLMTreeEmbeddingRegressor`) induce `num_trees` trees during `fit`, embed
  each sample as the concatenated truth values of every tree's inner nodes
  (off-path nodes count as `False`), optionally append preprocessed raw
  features (`append_raw_features=True`: imputation, one-hot encoding for
  nominal columns, MinMax scaling for numeric columns), and fit a downstream
  model on the embedding: one of the presets `estimator="mlp" | "hgbdt" | "lr"`
  (original paper parameter grids, `GridSearchCV`), or a user-supplied
  `downstream_estimator` (fitted directly, no grid). `transform(X)` returns
  the embedding matrix; `predict(X)` runs the fitted downstream model on it.

### Complexity constraints: `max_tree_depth` and `num_features`

Both are `None` (no constraint) or a positive integer ≥ 1; invalid values
(`0`, negative, non-int, bool) raise `ValueError` at `fit`. The two are
independent and can be combined. After generation, each tree is structurally
validated (parsed from its Python source): depth above `max_tree_depth` or
more distinct referenced features than `num_features` fails the attempt,
which is retried up to `num_retry_llm` times in total.

### Classification vs. regression

- Classification estimators map the tree's predictions back onto the original
  target labels seen at fit (`classes_`); a prediction is accepted if it
  matches a class label or is an integral position index into `classes_`
  (mirroring the "target value encoding" prompts).
- Regression estimators require numeric, finite predictions from the tree.

## Examples

These examples use small in-memory DataFrames with the feature names and
descriptions of the ACL injury dataset from the paper and dummy values.

### Classification induction

```python
import pandas as pd
from llm_trees import LLMTreeClassifier

X = pd.DataFrame(
    {
        "ccMF.S.T2.Me": [35.0, 45.0, 50.0, 30.0, 42.0, 48.0],
        "ccMF.D.T2.Me": [25.0, 35.0, 40.0, 22.0, 33.0, 38.0],
        "ccmfthctabame": [2.0, 3.0, 1.5, 2.8, 1.2, 2.2],
    }
)
y = pd.Series([1, 0, 1, 0, 0, 1], name="x1")  # 0 = normal, 1 = abnormal

model = LLMTreeClassifier(
    target={"x1": "cartilage thickness change in the central medial femur over 12 months following anterior cruciate ligament injury (0 = normal, 1 = abnormal)"},
    features={
        "ccMF.S.T2.Me": "baseline mean T2 values for the superficial cartilage layers in the central medial femur (ms)",
        "ccMF.D.T2.Me": "baseline mean T2 values for the deep cartilage layers in the central medial femur (ms)",
        "ccmfthctabame": "baseline mean cartilage thickness in the central medial femur (mm)",
    },
    max_tree_depth=2,
).fit(X, y)

print(model.predict(X))        # the 0/1 target labels
print(model.predict_proba(X))  # leaf class fractions
print(model.support_)          # features referenced by the induced tree
```

### Regression induction

```python
import pandas as pd
from llm_trees import LLMTreeRegressor

X = pd.DataFrame(
    {
        "ccMF.S.T2.Me": [35.0, 45.0, 50.0, 30.0, 42.0, 48.0],
        "ccMF.D.T2.Me": [25.0, 35.0, 40.0, 22.0, 33.0, 38.0],
    }
)
y = pd.Series([2.0, 3.0, 1.5, 2.8, 1.2, 2.2], name="ccmfthctabame")

model = LLMTreeRegressor(
    target={"ccmfthctabame": "baseline mean cartilage thickness in the central medial femur (mm)"},
    features={
        "ccMF.S.T2.Me": "baseline mean T2 values for the superficial cartilage layers in the central medial femur (ms)",
        "ccMF.D.T2.Me": "baseline mean T2 values for the deep cartilage layers in the central medial femur (ms)",
    },
    max_tree_depth=2,
).fit(X, y)

print(model.predict(X))  # float64 array of shape (n,)
print(model.support_)    # features referenced by the induced tree
```

### Embedding classification

```python
import pandas as pd
from llm_trees import LLMTreeEmbeddingClassifier

X = pd.DataFrame(
    {
        "ccMF.S.T2.Me": [35.0, 45.0, 50.0, 30.0, 42.0, 48.0],
        "ccMF.D.T2.Me": [25.0, 35.0, 40.0, 22.0, 33.0, 38.0],
        "ccmfthctabame": [2.0, 3.0, 1.5, 2.8, 1.2, 2.2],
    }
)
y = pd.Series([1, 0, 1, 0, 0, 1], name="x1")  # 0 = normal, 1 = abnormal

model = LLMTreeEmbeddingClassifier(
    target={"x1": "cartilage thickness change in the central medial femur over 12 months following anterior cruciate ligament injury (0 = normal, 1 = abnormal)"},
    features={
        "ccMF.S.T2.Me": "baseline mean T2 values for the superficial cartilage layers in the central medial femur (ms)",
        "ccMF.D.T2.Me": "baseline mean T2 values for the deep cartilage layers in the central medial femur (ms)",
        "ccmfthctabame": "baseline mean cartilage thickness in the central medial femur (mm)",
    },
    num_trees=2,
    estimator="lr",
).fit(X, y)

print(model.transform(X).shape)  # (6, 3 raw features + num inner nodes of the induced trees)
print(model.predict(X))          # prediction of the fitted downstream model
```

### Embedding regression

```python
import pandas as pd
from llm_trees import LLMTreeEmbeddingRegressor

X = pd.DataFrame(
    {
        "Height": [165.5, 178.2, 160.0, 182.4, 171.9, 168.3, 175.6],
        "Age": [24, 31, 27, 35, 29, 26, 41],
    }
)
y = pd.Series([22.4, 23.1, 25.8, 20.9, 23.6, 24.4, 21.7], name="BMI_FP")

model = LLMTreeEmbeddingRegressor(
    target={"BMI_FP": "body mass index (kg/m^2)"},
    features={
        "Height": "height (cm)",
        "Age": "age (years)",
    },
    num_trees=3,
    estimator="lr",
).fit(X, y)

print(model.transform(X).shape)  # (7, 2 raw features + num inner nodes of the induced trees)
print(model.predict(X))          # prediction of the fitted downstream model
```

### Two-step workflow

Every tree is generated by the same two-step dialogue as in the paper:

1. **Step 1** asks the LLM to induce a human-readable decision tree; the
   target is rendered by its description, every feature as
   `name: description`, optionally with the complexity constraints.
2. **Step 2** sends the step-1 response back as assistant context together
   with a follow-up prompt asking for a Python `predict(X)` function that
   returns `(prediction, nodes)` — the prediction plus the truth values of
   the inner nodes.

The two responses are concatenated, the Python function is extracted and
compiled, and the result is validated on the fit rows: every prediction must
be mappable to the target (original labels / numeric), every node vector must
be a one-dimensional sequence of consistent width (mappings and sets are
rejected), and the tree must satisfy `max_tree_depth` / `num_features`. The
failure classes are handled differently:

- **Compilation and runtime failures** — syntax errors, exceptions during
  the row-wise validation, unsupported node-vector containers or values,
  invalid labels, inconsistent node widths, constraint violations — trigger a
  retry.
- **Structural parse failures** — the function compiles and runs but does not
  follow the expected tree form, so its structure cannot be inspected —
  trigger a retry only when a complexity constraint is set, because the
  limits cannot be verified without a parsed structure. Without constraints,
  such an executable-but-unparsable function is accepted as-is: `predict`
  works, but the structural attributes (`tree_`, `support_`, `sparsity_`) are
  unavailable and `LLMTreeClassifier.predict_proba` raises `RuntimeError`.

After `num_retry_llm` total failed attempts, `fit` raises `ValueError`.

## Testing

The test suite is fully offline: no API keys are needed and no test calls a
real model. LLM responses are scripted, deterministic fakes, so the suite
exercises the prompt construction, generation, validation, and prediction
pipeline around a synthetic two-step dialogue. From the repository root,
run:

```sh
pip install -e .[test]
python -m pytest tests/ -q
```

## License

This project is licensed under the [MIT License](https://github.com/ml-lab-htw/llm-trees/blob/main/LICENSE). See the `LICENSE` file for details.

## Authors

- Ricardo Knauer (HTW Berlin)
- Mario Koddenbrock (HTW Berlin)
