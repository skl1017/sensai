"""Unit tests for `FewShotSection` (#32 part 1)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from core.types import Context
from nodes.fewshot_section import FewShotSection
from tests.fakes import FakeEmbedder, FakeLLM

_BANK = {
    "examples": [
        {
            "question": "What is the derivative of x²",
            "answer": "The derivative of x² is 2x.",
        },
        {
            "question": "How to write an email",
            "answer": "Here is an email template.",
        },
        {
            "question": "Pythagorean theorem triangle",
            "answer": "c² = a² + b²",
        },
        {
            "question": "How to add two fractions",
            "answer": "Put them over a common denominator.",
        },
        {
            "question": "How to compute a percentage",
            "answer": "Multiply by the rate as a decimal.",
        },
    ]
}


def write_bank(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def make_ctx(user_input: str = "hi", state: dict | None = None) -> Context:
    return Context(
        user_input=user_input,
        messages=[],
        llm=FakeLLM([]),
        tools={},
        state=state or {},
    )


class _Deps:
    def __init__(self, embedder, config_dir: Path) -> None:
        self.embedder = embedder
        self.config_dir = config_dir


# --- relevance / top-k -------------------------------------------------------------


async def test_most_relevant_example_is_injected_first(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=1)

    text = await section.build(make_ctx("What is the derivative of x³"))

    expected_block = "Q: What is the derivative of x²\nA: The derivative of x² is 2x."
    assert text == f"Examples of good answers:\n\n{expected_block}"


async def test_k_is_respected(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=2)

    text = await section.build(make_ctx("What is the derivative of x³"))

    assert text.count("Q:") == 2


async def test_min_score_filters_out_everything_returns_none(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=3, min_score=1.1)  # unreachable score

    text = await section.build(make_ctx("What is the derivative of x³"))

    assert text is None


async def test_empty_bank_returns_none(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", {"examples": []})
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path)

    text = await section.build(make_ctx("bonjour"))

    assert text is None


# --- query source ------------------------------------------------------------------


async def test_state_query_takes_precedence_over_user_input(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=1)

    ctx = make_ctx(
        user_input="How to write a professional email",
        state={"query": "What is the derivative of x³"},
    )
    text = await section.build(ctx)

    assert "derivative" in text
    assert "email" not in text


# --- lazy single embedding of the bank ----------------------------------------------


async def test_bank_is_embedded_only_once_across_sequential_builds(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=1)

    await section.build(make_ctx("What is the derivative of x³"))
    await section.build(make_ctx("How to add two fractions"))

    bank_calls = [call for call in embedder.calls if len(call) == len(_BANK["examples"])]
    assert len(bank_calls) == 1
    assert len(embedder.calls) == 3  # 1 bank embed + 2 query embeds


async def test_bank_is_embedded_only_once_across_concurrent_builds(tmp_path):
    path = write_bank(tmp_path / "fewshot.yaml", _BANK)
    embedder = FakeEmbedder()
    section = FewShotSection(embedder, path, k=1)

    await asyncio.gather(
        section.build(make_ctx("What is the derivative of x³")),
        section.build(make_ctx("How to add two fractions")),
    )

    bank_calls = [call for call in embedder.calls if len(call) == len(_BANK["examples"])]
    assert len(bank_calls) == 1


# --- validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        ["not", "a", "mapping"],
        {"not_examples": []},
        {"examples": "not a list"},
        {"examples": ["not a dict"]},
        {"examples": [{"question": "Q?"}]},
        {"examples": [{"answer": "A."}]},
        {"examples": [{"question": "", "answer": "A."}]},
        {"examples": [{"question": "Q?", "answer": "  "}]},
        {"examples": [{"question": 1, "answer": "A."}]},
    ],
)
async def test_invalid_bank_raises_value_error(tmp_path, data):
    path = write_bank(tmp_path / "fewshot.yaml", data)
    section = FewShotSection(FakeEmbedder(), path)

    with pytest.raises(ValueError):
        await section.build(make_ctx("bonjour"))


# --- from_config ---------------------------------------------------------------------


def test_from_config_missing_embedder_raises():
    with pytest.raises(ValueError, match="embedder"):
        FewShotSection.from_config({}, _Deps(None, Path("config")))


def test_from_config_unknown_param_raises():
    with pytest.raises(ValueError):
        FewShotSection.from_config({"bogus": 1}, _Deps(FakeEmbedder(), Path("config")))


def test_from_config_defaults_path_to_config_dir():
    deps = _Deps(FakeEmbedder(), Path("config"))
    section = FewShotSection.from_config({}, deps)

    assert section.path == Path("config") / "fewshot.yaml"
    assert section.k == 3


def test_from_config_passes_k_and_min_score():
    deps = _Deps(FakeEmbedder(), Path("config"))
    section = FewShotSection.from_config({"k": 5, "min_score": 0.4}, deps)

    assert section.k == 5
    assert section.min_score == 0.4


def test_from_config_custom_path():
    deps = _Deps(FakeEmbedder(), Path("config"))
    section = FewShotSection.from_config({"path": "other/bank.yaml"}, deps)

    assert section.path == Path("other/bank.yaml")


# --- real bank -------------------------------------------------------------------------


async def test_real_fewshot_yaml_loads_and_builds():
    section = FewShotSection(FakeEmbedder(), Path("config/fewshot.yaml"), k=3)

    text = await section.build(make_ctx("What is the derivative of x^3?"))

    assert text is not None
    assert text.startswith("Examples of good answers:")
    assert text.count("Q:") == 3

