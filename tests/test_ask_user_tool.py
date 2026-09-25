"""Unit tests for `AskUserTool` (X4, wiki §8.6)."""

from __future__ import annotations

import asyncio

from tests.fakes import FakeUI
from tools.ask_user_tool import NO_ANSWER, AskUserTool


class _SlowUI:
    """`UserInterface` double whose `ask` never returns before the caller times out."""

    async def ask(self, question: str, kind: str = "confirm", options: list[str] | None = None):
        await asyncio.sleep(1)
        return "too late"


def test_declares_no_side_effect_and_owns_its_timeout():
    tool = AskUserTool(FakeUI())
    assert tool.name == "ask_user"
    assert "don't guess" in tool.description.lower() or "guess" in tool.description.lower()
    assert tool.enforces_own_timeout is True


async def test_text_kind_is_the_default_and_returns_the_answer():
    ui = FakeUI(["Marin"])
    result = await AskUserTool(ui).run(question="Quel est ton prénom ?")
    assert result == "Marin"
    question, kind, options = ui.questions[0]
    assert (question, kind, options) == ("Quel est ton prénom ?", "text", None)


async def test_confirm_kind_returns_yes_or_no_verbatim():
    ui = FakeUI(["no"])
    result = await AskUserTool(ui).run(question="Confirmer ?", kind="confirm")
    assert result == "no"
    _question, kind, _options = ui.questions[0]
    assert kind == "confirm"


async def test_choice_kind_passes_options_through_and_returns_the_pick():
    ui = FakeUI(["vert"])
    options = ["rouge", "vert", "bleu"]
    result = await AskUserTool(ui).run(question="Quelle couleur ?", kind="choice", options=options)
    assert result == "vert"
    question, kind, sent_options = ui.questions[0]
    assert (question, kind, sent_options) == ("Quelle couleur ?", "choice", options)


async def test_choice_kind_without_options_raises():
    tool = AskUserTool(FakeUI())
    try:
        await tool.run(question="Quelle couleur ?", kind="choice")
    except ValueError as exc:
        assert "options" in str(exc)
        return
    raise AssertionError("expected ValueError")


async def test_unsupported_kind_raises():
    tool = AskUserTool(FakeUI())
    try:
        await tool.run(question="??", kind="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
        return
    raise AssertionError("expected ValueError")


async def test_no_timeout_waits_indefinitely_for_the_answer():
    ui = FakeUI(["ok"])
    result = await AskUserTool(ui, timeout=None).run(question="q?")
    assert result == "ok"


async def test_timeout_elapsed_returns_no_answer_instead_of_raising():
    tool = AskUserTool(_SlowUI(), timeout=0.01)
    result = await tool.run(question="q?")
    assert result == NO_ANSWER


def test_from_config_wires_ui_and_optional_timeout():
    class _Deps:
        ui = FakeUI()

    tool = AskUserTool.from_config({}, _Deps())
    assert tool.ui is _Deps.ui
    assert tool.timeout is None

    tool = AskUserTool.from_config({"timeout": 5}, _Deps())
    assert tool.timeout == 5
