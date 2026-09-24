"""Unit tests for `ApprovedTool` (wiki §8.9)."""

from __future__ import annotations

import asyncio

from tests.fakes import FakeTool, FakeUI
from tools.approved_tool import REFUSAL, ApprovedTool


def test_mirrors_inner_tool_and_owns_timeout():
    inner = FakeTool(name="write_file", description="d", parameters={"type": "object"})
    tool = ApprovedTool(inner, FakeUI())
    assert (tool.name, tool.description, tool.parameters) == ("write_file", "d", inner.parameters)
    assert tool.enforces_own_timeout is True


async def test_yes_runs_inner_tool_with_plan_shown():
    inner = FakeTool(name="write_file", result="written")
    ui = FakeUI(["yes"])
    result = await ApprovedTool(inner, ui).run(path="a.txt")
    assert result == "written"
    assert inner.calls == [{"path": "a.txt"}]
    question, kind = ui.questions[0]
    assert kind == "confirm"
    assert "write_file" in question and '"path": "a.txt"' in question


async def test_no_returns_refusal_without_running():
    inner = FakeTool()
    result = await ApprovedTool(inner, FakeUI(["no"])).run()
    assert result == REFUSAL
    assert inner.calls == []


async def test_any_other_answer_is_a_refusal():
    inner = FakeTool()
    assert await ApprovedTool(inner, FakeUI(["Yes please"])).run() == REFUSAL
    assert inner.calls == []


async def test_timeout_applies_only_after_approval():
    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    tool = ApprovedTool(FakeTool(result=slow), FakeUI(["yes"]), inner_timeout=0.01)
    try:
        await tool.run()
    except TimeoutError:
        return
    raise AssertionError("expected TimeoutError")
