"""Tests for the prompt_file parameter: generated step prompts written for inspection."""

import pandas as pd

from conftest import DIALOGUE_SEPARATOR, ScriptedClient
from llm_trees._llm_tree_classifier import LLMTreeClassifier
from llm_trees._llm_tree_embedding_classifier import LLMTreeEmbeddingClassifier

TARGET = {"x1": "synthetic system state (normal / abnormal)"}
FD = {"a": "feature a", "b": "feature b"}

CLS_CODE = """```python
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["a"] <= 1.0
    if nodes[0]:
        prediction = "normal"
    else:
        prediction = "abnormal"
    return prediction, nodes
```"""

RAW = "A decision tree.\n\n\n\n" + DIALOGUE_SEPARATOR + "\n\n\n" + CLS_CODE


def _X_y():
    X = pd.DataFrame({"a": [0.0, 2.0, 0.5, 1.5, 0.0, 2.5], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series(["normal", "abnormal", "normal", "abnormal", "normal", "abnormal"])
    return X, y


def _patch_client(monkeypatch, client, mod_name):
    monkeypatch.setattr(f"llm_trees.{mod_name}.make_client", lambda backend, temperature, seed: client)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_prompt_file_default_writes_prompt_txt(monkeypatch, tmp_path):
    X, y = _X_y()
    _patch_client(monkeypatch, ScriptedClient(RAW), "_llm_tree_classifier")
    LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter").fit(X, y)

    path = tmp_path / "prompt.txt"
    assert path.exists()
    content = _read(path)
    # both step prompts, in order, with inspectable section headers
    first_at = content.index("===== FIRST PROMPT (tree induction) =====")
    second_at = content.index("===== SECOND PROMPT (Python function generation) =====")
    assert first_at < second_at
    assert "The tree should predict the synthetic system state (normal / abnormal)." in content
    assert "Features: a: feature a, b: feature b" in content
    assert "Here are the exact keys of the features" in content
    assert "x1: synthetic system state (normal / abnormal)" in content.split("SECOND PROMPT")[1]


def test_prompt_file_custom_path_creates_parents(monkeypatch, tmp_path):
    X, y = _X_y()
    _patch_client(monkeypatch, ScriptedClient(RAW), "_llm_tree_classifier")
    out = tmp_path / "out" / "prompts" / "p.txt"
    LLMTreeClassifier(target=TARGET, features=FD, prompt_file=str(out), backend="openrouter").fit(X, y)
    assert out.exists()
    assert "Features: a: feature a, b: feature b" in _read(out)


def test_prompt_file_disabled(monkeypatch, tmp_path):
    X, y = _X_y()
    _patch_client(monkeypatch, ScriptedClient(RAW), "_llm_tree_classifier")
    LLMTreeClassifier(target=TARGET, features=FD, prompt_file=False, backend="openrouter").fit(X, y)
    assert not (tmp_path / "prompt.txt").exists()
    assert list(tmp_path.iterdir()) == []


def test_prompt_file_written_when_generation_fails(monkeypatch, tmp_path):
    # the prompts are written before the LLM calls, so a failed fit still
    # leaves an inspectable record of what was asked
    X, y = _X_y()
    _patch_client(monkeypatch, ScriptedClient("no code at all in this response"), "_llm_tree_classifier")
    try:
        LLMTreeClassifier(target=TARGET, features=FD, backend="openrouter", num_retry_llm=1).fit(X, y)
    except ValueError:
        pass
    else:
        raise AssertionError("fit was expected to fail")
    assert (tmp_path / "prompt.txt").exists()


def test_prompt_file_embedding_estimator(monkeypatch, tmp_path):
    X, y = _X_y()
    _patch_client(monkeypatch, ScriptedClient(RAW), "_llm_tree_embedding_classifier")
    LLMTreeEmbeddingClassifier(
        target=TARGET, features=FD, num_trees=2, estimator="lr",
        append_raw_features=False, backend="openrouter",
    ).fit(X, y)
    content = _read(tmp_path / "prompt.txt")
    assert content.count("===== FIRST PROMPT (tree induction) =====") == 1
    assert "Features: a: feature a, b: feature b" in content
