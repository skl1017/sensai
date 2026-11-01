"""Unit tests for `nodes/rag_filter_extractor.py` (#135)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from core.types import Message
from nodes.rag_filter_extractor import extract_filters
from tests.fakes import FakeLLM

FILES = ["cv.pdf", "rh.pdf"]
TODAY = date(2026, 11, 1)


def answer(file_name=None, date_from=None, date_to=None) -> str:
    return json.dumps({"file_name": file_name, "date_from": date_from, "date_to": date_to})


async def extract(script, files=FILES):
    llm = FakeLLM(script)
    return await extract_filters(llm, "Question ?", files, today=TODAY), llm


async def test_question_without_restriction_gives_no_filter():
    filters, _ = await extract([answer()])

    assert filters == {}


async def test_dates_and_indexed_file_become_filters():
    filters, _ = await extract([answer("rh.pdf", "2024-01-01", "2024-12-31")])

    assert filters == {
        "file_name": "rh.pdf",
        "date_from": date(2024, 1, 1),
        "date_to": date(2024, 12, 31),
    }


async def test_prompt_gives_today_files_and_question():
    _, llm = await extract([answer()])

    prompt = llm.calls[0]["messages"][-1].content
    assert "Today: 2026-11-01" in prompt
    assert "- rh.pdf" in prompt
    assert "Question ?" in prompt


async def test_file_that_is_not_indexed_is_dropped():
    filters, _ = await extract([answer("invente.pdf", None, "2023-12-31")])

    assert filters == {"date_to": date(2023, 12, 31)}


@pytest.mark.parametrize("bad", ["hier", "2023", 2023])
async def test_unreadable_date_is_dropped(bad):
    filters, _ = await extract([answer("rh.pdf", bad, None)])

    assert filters == {"file_name": "rh.pdf"}


async def test_reversed_dates_are_dropped():
    filters, _ = await extract([answer(None, "2025-01-01", "2024-01-01")])

    assert filters == {}


async def test_output_that_is_not_an_object_gives_no_filter():
    filters, _ = await extract(["[]"])

    assert filters == {}


async def test_invalid_json_gives_no_filter():
    filters, llm = await extract(["pas du json", "toujours pas"])

    assert filters == {}
    assert len(llm.calls) == 2


async def test_llm_failure_gives_no_filter(caplog):
    class BrokenLLM(FakeLLM):
        async def chat(self, messages: list[Message], tools=None, schema=None):
            raise RuntimeError("secret question text")
            yield

    filters = await extract_filters(BrokenLLM([]), "Question ?", FILES, today=TODAY)

    assert filters == {}
    assert "secret question text" not in caplog.text


async def test_no_indexed_file_skips_the_llm():
    filters, llm = await extract([], files=[])

    assert filters == {}
    assert llm.calls == []
