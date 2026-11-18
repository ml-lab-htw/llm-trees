"""Tests for llm_trees._llm: the OpenRouter two-step client."""

import importlib.util
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.fixtures import DIALOGUE_SEPARATOR_FIXTURE
from llm_trees import _llm
from llm_trees._llm import OpenRouterClient, make_client
from llm_trees._tree import DIALOGUE_SEPARATOR


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    yield


def _capture_create(captured):
    # snapshot kwargs at call time: the messages list is mutated between calls
    def create(**kwargs):
        captured.append({**kwargs, "messages": list(kwargs["messages"])})
        return _response("R1") if len(captured) == 1 else _response("R2")
    return create


def test_two_step_message_flow_preserves_context(openrouter_env):
    captured = []
    with mock.patch.object(_llm, "OpenAI") as mock_openai:
        client_instance = mock_openai.return_value
        client_instance.chat.completions.create.side_effect = _capture_create(captured)
        client = OpenRouterClient(temperature=0.0, seed=0)
        result = client.two_step("FIRST_PROMPT", "SECOND_PROMPT")

    assert len(captured) == 2
    first_kwargs = captured[0]
    second_kwargs = captured[1]
    assert first_kwargs["messages"] == [{"role": "user", "content": "FIRST_PROMPT"}]
    # step 2 must retain the context and response of step 1
    assert second_kwargs["messages"] == [
        {"role": "user", "content": "FIRST_PROMPT"},
        {"role": "assistant", "content": "R1"},
        {"role": "user", "content": "SECOND_PROMPT"},
    ]
    assert first_kwargs["model"] == "test-model"
    assert second_kwargs["model"] == "test-model"
    assert result == "R1\n\n\n" + "#" * 25 + "\n\n\nR2"
    # the production separator and the fixture used across the test suite agree
    assert DIALOGUE_SEPARATOR == DIALOGUE_SEPARATOR_FIXTURE


def test_temperature_and_seed_forwarded(openrouter_env):
    captured = []
    with mock.patch.object(_llm, "OpenAI") as mock_openai:
        client_instance = mock_openai.return_value
        client_instance.chat.completions.create.side_effect = _capture_create(captured)
        client = OpenRouterClient(temperature=1.0, seed=7)
        client.two_step("f", "s")

    for kwargs in captured:
        assert kwargs["temperature"] == 1.0
        assert kwargs["seed"] == 7
    # client is constructed against the OpenRouter endpoint with the env key
    mock_openai.assert_called_once_with(
        base_url=_llm.OPENROUTER_BASE_URL, api_key="test-key"
    )


def test_base_url_can_point_at_another_openai_compatible_server(openrouter_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:8000/v1")
    with mock.patch.object(_llm, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _response("R")
        OpenRouterClient(temperature=0.0, seed=0).two_step("f", "s")
    mock_openai.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="test-key")


def test_missing_openrouter_env_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")
    client = OpenRouterClient(temperature=0.0, seed=0)  # construction must not require the key
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        client.two_step("f", "s")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        client.two_step("f", "s")


def test_env_file_discovered_from_cwd(monkeypatch, tmp_path):
    # a project-local .env in the CWD must be found even when the package is
    # installed elsewhere (site-packages): discovery must be CWD-based, not
    # based on the package file location. A fresh module object runs the
    # same module-level code without touching the imported module's state.
    key = "sk-test-env-discovery-000"
    (tmp_path / ".env").write_text(f"OPENROUTER_API_KEY={key}\nOPENROUTER_MODEL=test-model\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel")  # so teardown restores the original state
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("llm_trees._llm_env_probe", _llm.__file__)
    fresh = importlib.util.module_from_spec(spec)
    fresh.__package__ = "llm_trees"
    spec.loader.exec_module(fresh)
    assert os.environ.get("OPENROUTER_API_KEY") == key
    assert os.environ.get("OPENROUTER_MODEL") == "test-model"


def test_make_client_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        make_client("dummy", 0.0, 0)
    assert isinstance(make_client("openrouter", 0.0, 0), OpenRouterClient)
