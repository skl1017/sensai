"""Integration tests against a real, local Ollama server.

Skipped (not failed) when Ollama is unreachable or the required models are
not pulled. Run explicitly with `pytest -m slow tests/test_ollama_integration.py`.
"""

from __future__ import annotations

import os

import httpx
import pytest

from core.llm import LLMModelNotFound, chat_json
from core.types import Message

pytestmark = pytest.mark.slow

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEST_MODEL = os.environ.get("SENSAI_TEST_MODEL", "qwen2.5:0.5b")
TEST_EMBED_MODEL = os.environ.get("SENSAI_TEST_EMBED_MODEL", "nomic-embed-text")


def _installed_models() -> set[str] | None:
    """Return the set of installed model names, or None if Ollama is unreachable."""
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    return {m["name"] for m in resp.json().get("models", [])}


def _is_installed(model: str, models: set[str]) -> bool:
    """Ollama lists untagged models as `<name>:latest`."""
    return model in models or f"{model}:latest" in models


@pytest.fixture(scope="module")
def ollama_available():
    models = _installed_models()
    if models is None:
        pytest.skip(f"Ollama not reachable at {OLLAMA_HOST}")
    if not _is_installed(TEST_MODEL, models):
        pytest.skip(f"model {TEST_MODEL!r} not pulled in Ollama")
    return models


@pytest.fixture(scope="module")
def embed_model_available(ollama_available):
    if not _is_installed(TEST_EMBED_MODEL, ollama_available):
        pytest.skip(f"embed model {TEST_EMBED_MODEL!r} not pulled in Ollama")


@pytest.fixture
async def llm(ollama_available):
    from llm.ollama import Ollama

    async with Ollama(model=TEST_MODEL, host=OLLAMA_HOST, options={"temperature": 0}) as client:
        yield client


async def test_streaming_yields_chunks_with_final_token_counts(llm):
    messages = [Message(role="user", content="Say hello in one short sentence.")]

    chunks = []
    async for chunk in llm.chat(messages):
        chunks.append(chunk)

    assert len(chunks) >= 2
    last = chunks[-1]
    assert last.done is True
    assert isinstance(last.prompt_tokens, int)
    assert isinstance(last.completion_tokens, int)

    text = "".join(c.text for c in chunks)
    assert text.strip() != ""


async def test_tool_call_round_trip(llm):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [Message(role="user", content="What's the weather like in Paris?")]

    call = None
    async for chunk in llm.chat(messages, tools=tools):
        for tc in chunk.tool_calls:
            call = tc

    assert call is not None
    assert call.name == "get_weather"
    assert call.id

    messages.append(Message(role="assistant", content="", tool_calls=[call]))
    messages.append(
        Message(role="tool", content="22°C sunny", tool_call_id=call.id, name="get_weather")
    )

    final_text = ""
    async for chunk in llm.chat(messages, tools=tools):
        final_text += chunk.text

    assert final_text.strip() != ""


async def test_chat_json_returns_dict_with_required_keys(llm):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    messages = [Message(role="user", content='Reply with JSON: {"answer": "hi"}')]

    result = await chat_json(llm, messages, schema)

    assert isinstance(result, dict)
    assert "answer" in result


async def test_embed_returns_vectors(embed_model_available):
    from llm.ollama import OllamaEmbedder

    async with OllamaEmbedder(model=TEST_EMBED_MODEL, host=OLLAMA_HOST) as client:
        vectors = await client.embed(["hello world", "goodbye world"])

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])


async def test_missing_model_raises_model_not_found(ollama_available):
    from llm.ollama import Ollama

    async with Ollama(model="sensai-does-not-exist:latest", host=OLLAMA_HOST) as client:
        with pytest.raises(LLMModelNotFound):
            async for _ in client.chat([Message(role="user", content="hi")]):
                pass
