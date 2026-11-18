"""Public API and packaging tests."""

import importlib.util
import re
from pathlib import Path

import llm_trees

ROOT = Path(__file__).parent.parent


def test_public_imports():
    from llm_trees import (
        LLMTreeClassifier,
        LLMTreeRegressor,
        LLMTreeEmbeddingClassifier,
        LLMTreeEmbeddingRegressor,
    )

    assert llm_trees.__all__ == [
        "LLMTreeClassifier",
        "LLMTreeRegressor",
        "LLMTreeEmbeddingClassifier",
        "LLMTreeEmbeddingRegressor",
    ]


def test_version_030_synced():
    assert llm_trees.__version__ == "0.3.0"
    setup_src = (ROOT / "setup.py").read_text()
    m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", setup_src)
    assert m and m.group(1) == llm_trees.__version__


def test_no_legacy_modules_importable():
    for mod in [
        "llm_trees.cli",
        "llm_trees.config",
        "llm_trees.llms",
        "llm_trees.io",
        "llm_trees.metrics",
        "llm_trees.embeddings",
        "llm_trees.induction_utils",
        "llm_trees.embedding_utils",
        "llm_trees.prompts",
        "llm_trees.utils",
        "llm_trees.prompt_examples",
    ]:
        assert importlib.util.find_spec(mod) is None, f"legacy module still importable: {mod}"


def test_no_removed_provider_dependencies_imported():
    pkg_dir = Path(llm_trees.__file__).parent
    forbidden = [
        "anthropic",
        "vertexai",
        "xgboost",
        "generate_gpt_tree",
        "generate_claude_tree",
        "generate_gemini_tree",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
    ]
    for path in sorted(pkg_dir.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in src, f"{token!r} found in {path}"


def test_setup_metadata():
    setup_src = (ROOT / "setup.py").read_text()
    assert "entry_points" not in setup_src
    for dep in [
        "'pandas~=2.2.3'",
        "'scikit-learn>=1.7,<2'",
        "'numpy>=1.26,<3'",
        "'openai~=1.57.2'",
        "'python-dotenv~=1.2.2'",
    ]:
        assert dep in setup_src, dep
    assert "extras_require" in setup_src
    assert "'test': ['pytest']" in setup_src
    assert "python_requires='>=3.10'" in setup_src
    assert (
        "'Zero-shot decision tree induction and embedding using large language models'"
        in setup_src
    )
    # removed providers must not be dependencies
    for dep in ["anthropic", "google-cloud-aiplatform", "xgboost", "matplotlib", "seaborn"]:
        assert dep not in setup_src, dep


def test_requirements_txt():
    req = (ROOT / "requirements.txt").read_text()
    for dep in ["pandas~=2.2.3", "scikit-learn>=1.7,<2", "numpy>=1.26,<3", "openai~=1.57.2", "python-dotenv~=1.2.2", "pytest"]:
        assert dep in req, dep
    for dep in ["anthropic", "xgboost", "matplotlib", "seaborn", "tqdm", "cachetools", "prettytable"]:
        assert dep not in req, dep


def test_gitignore_without_dataset_lines():
    gitignore = (ROOT / ".gitignore").read_text()
    assert "/data_sets/" not in gitignore
