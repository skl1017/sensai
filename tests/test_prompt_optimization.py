"""Prompt-optimization bench: few-shot top-1 hit rate and query-rewrite lift (#32/#97).

Slow, network-touching measurement against a real, local Ollama server (same
models as `config/agent.yaml`): skipped (not failed) when Ollama is
unreachable or the required models aren't pulled -- see
`tests/test_ollama_integration.py`, whose skip pattern this reuses. Run
explicitly with `pytest -m slow tests/test_prompt_optimization.py -s` (`-s`
so the printed hit rates aren't swallowed).

Two measurements:

- `test_fewshot_top1_hit_rate`: for a handful of paraphrases of
  `config/fewshot.yaml` bank questions, does `FewShotSection` (`k=1`, real
  `OllamaEmbedder`) pick the paraphrased entry as its closest match?
- `test_query_rewrite_improves_followup_hit_rate`: for a few pronoun-heavy
  follow-ups ("et pour 300 ?" style), compares that same top-1 hit rate
  using the raw follow-up as the query versus `rewrite_query`'s standalone
  rewrite (real `Ollama` chat). The only hard assertion is that rewriting
  never hurts retrieval on this set (`with_rewrite_hits >= without_rewrite_hits`);
  the numbers themselves are printed for a human to track over time.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from core.types import Context, Message
from nodes.fewshot_section import FewShotSection
from nodes.query_rewriter import rewrite_query
from tests.fakes import FakeLLM

pytestmark = pytest.mark.slow

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEST_MODEL = os.environ.get("SENSAI_TEST_MODEL", "qwen2.5:0.5b")
TEST_EMBED_MODEL = os.environ.get("SENSAI_TEST_EMBED_MODEL", "nomic-embed-text")
FEWSHOT_PATH = Path("config/fewshot.yaml")


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


@pytest.fixture
async def embedder(embed_model_available):
    from llm.ollama import OllamaEmbedder

    async with OllamaEmbedder(model=TEST_EMBED_MODEL, host=OLLAMA_HOST) as client:
        yield client


def _make_ctx(query: str) -> Context:
    return Context(query, [], FakeLLM([]), {}, state={"query": query})


async def _top1_question(section: FewShotSection, query: str) -> str | None:
    """The bank question `section` (`k=1`) picks as its closest match to `query`, or `None`."""
    text = await section.build(_make_ctx(query))
    if text is None:
        return None
    # "Examples of good answers:\n\nQ: <question>\nA: <answer>"
    for line in text.splitlines():
        if line.startswith("Q: "):
            return line[len("Q: ") :]
    return None


# --- top-1 hit rate: paraphrases of bank questions -> the entry they paraphrase --------

_TOP1_CASES = [
    ("Comment calcule-t-on la dérivée de x² + 3x ?", "Quelle est la dérivée de x² + 3x ?"),
    ("Comment simplifier 18 sur 24 ?", "Comment simplifier la fraction 18/24 ?"),
    (
        "Un triangle rectangle a des côtés de 3 et 4 cm, quelle longueur fait l'hypoténuse ?",
        "Un triangle rectangle a des côtés de 3 cm et 4 cm. Quelle est l'hypoténuse ?",
    ),
    ("Comment on lit un fichier CSV avec Python ?", "Comment lire un fichier CSV en Python ?"),
    ("Résous 2x + 5 = 13.", "Résous l'équation 2x + 5 = 13."),
]


async def test_fewshot_top1_hit_rate(embedder):
    section = FewShotSection(embedder, FEWSHOT_PATH, k=1)

    hits = 0
    for query, expected in _TOP1_CASES:
        got = await _top1_question(section, query)
        if got == expected:
            hits += 1

    rate = hits / len(_TOP1_CASES)
    print(f"\nfew-shot top-1 hit rate: {hits}/{len(_TOP1_CASES)} ({rate:.0%})")


# --- query rewrite lift on pronoun-heavy follow-ups -------------------------------------

_FOLLOWUP_CASES = [
    (
        [
            Message("user", "Comment calculer 20 % de 150 ?"),
            Message("assistant", "20 % de 150 = 150 × 0,20 = 30."),
        ],
        "Et pour l'autre valeur ?",
        "Comment calculer 20 % de 150 ?",
    ),
    (
        [
            Message("user", "Comment simplifier la fraction 18/24 ?"),
            Message("assistant", "Le PGCD de 18 et 24 est 6, donc 18/24 = 3/4."),
        ],
        "Et celle-là ?",
        "Comment simplifier la fraction 18/24 ?",
    ),
    (
        [
            Message(
                "user",
                "Un triangle rectangle a des côtés de 3 cm et 4 cm. Quelle est l'hypoténuse ?",
            ),
            Message("assistant", "D'après Pythagore : hypoténuse = √(3² + 4²) = 5 cm."),
        ],
        "Et pour le deuxième triangle ?",
        "Un triangle rectangle a des côtés de 3 cm et 4 cm. Quelle est l'hypoténuse ?",
    ),
]


async def test_query_rewrite_improves_followup_hit_rate(embedder, llm):
    section = FewShotSection(embedder, FEWSHOT_PATH, k=1)

    without_hits = 0
    with_hits = 0
    for history, followup, expected in _FOLLOWUP_CASES:
        got_without = await _top1_question(section, followup)
        if got_without == expected:
            without_hits += 1

        rewritten = await rewrite_query(llm, history, followup)
        got_with = await _top1_question(section, rewritten)
        if got_with == expected:
            with_hits += 1

    total = len(_FOLLOWUP_CASES)
    print(
        f"\nfollow-up hit rate without rewrite: {without_hits}/{total} ({without_hits / total:.0%})"
    )
    print(f"follow-up hit rate with rewrite:    {with_hits}/{total} ({with_hits / total:.0%})")

    assert with_hits >= without_hits
