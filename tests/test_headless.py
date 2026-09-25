"""Unit tests for `HeadlessUI` (A3, wiki §8.6/8.9): deny by default, no user to ask."""

from __future__ import annotations

from core.ui import UserInterface
from tests.fakes import FakeTool
from tools.approved_tool import REFUSAL, ApprovedTool
from tools.ask_user_tool import AskUserTool
from ui.headless import NO_USER, HeadlessUI


def test_satisfies_the_userinterface_protocol():
    assert isinstance(HeadlessUI(), UserInterface)


def test_on_token_and_on_event_are_no_ops():
    ui = HeadlessUI()
    assert ui.on_token("hello") is None
    assert ui.on_event("tool_call", {"name": "x", "args": {}}) is None


async def test_confirm_is_always_refused():
    assert await HeadlessUI().ask("Autoriser ?", kind="confirm") == "no"


async def test_text_and_choice_report_no_user_available():
    ui = HeadlessUI()
    assert await ui.ask("Quel est ton prénom ?", kind="text") == NO_USER
    assert await ui.ask("Rouge ou vert ?", kind="choice", options=["rouge", "vert"]) == NO_USER


# --- acceptance criterion: forgetting to wire a UI never makes the agent dangerous ---


async def test_approved_tool_never_runs_the_inner_tool_headless():
    inner = FakeTool(name="write_file", result="written")
    result = await ApprovedTool(inner, HeadlessUI()).run(path="a.txt")
    assert result == REFUSAL
    assert inner.calls == []


async def test_ask_user_tool_never_hangs_headless():
    result = await AskUserTool(HeadlessUI()).run(question="Quel est ton prénom ?")
    assert result == NO_USER
