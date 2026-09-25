"""Unit tests for `apply_budget` (#31, wiki M2: token budget + rolling summary)."""

from __future__ import annotations

from core.types import Message
from nodes.context_budget import _digest, apply_budget
from tests.fakes import FakeLLM


def make_turn(user_text: str, assistant_text: str) -> list[Message]:
    return [Message("user", user_text), Message("assistant", assistant_text)]


def flatten(turns: list[list[Message]]) -> list[Message]:
    return [message for turn in turns for message in turn]


# --- under budget --------------------------------------------------------------------


async def test_under_budget_leaves_messages_unchanged_and_writes_stats():
    system = Message("system", "You are helpful.")
    history = flatten([make_turn("hi", "hello")])
    current = Message("user", "how are you?")
    messages = [system, *history, current]
    original = list(messages)

    llm = FakeLLM(script=["unused"], max_context_tokens=8192)
    state: dict = {}

    result = await apply_budget(messages, llm, state)

    assert [(m.role, m.content) for m in result] == [(m.role, m.content) for m in messages]
    assert result is not messages
    assert messages == original
    assert llm.calls == []

    stats = state["context_stats"]
    assert stats["compressed"] is False
    assert stats["tokens_before"] == stats["tokens_after"]
    assert "summary" not in state


async def test_input_list_and_messages_not_mutated_when_under_budget():
    system = Message("system", "Persona.")
    history = flatten([make_turn("q", "a")])
    current = Message("user", "current")
    messages = [system, *history, current]
    original_ids = [id(m) for m in messages]
    original_contents = [m.content for m in messages]

    llm = FakeLLM(script=[], max_context_tokens=8192)
    await apply_budget(messages, llm, {})

    assert [id(m) for m in messages] == original_ids
    assert [m.content for m in messages] == original_contents


# --- over budget: summarize old turns, keep the rest intact --------------------------


_Conversation = tuple[list[Message], Message, list[list[Message]], Message]


def _build_over_budget_conversation() -> _Conversation:
    old1 = make_turn(
        "Client call: my name is Mirabelle Kovacs, account number 4817.",
        "Got it, I will note that down for the file.",
    )
    old2 = make_turn(
        "What did we decide about the storage layer?",
        "We discussed it at length and the decision is: on garde PostgreSQL.",
    )
    kept = [make_turn(f"recent question {i}", f"recent answer {i}") for i in range(4)]
    system = Message("system", "You are a helpful assistant.")
    current = Message("user", "current question")
    history = old1 + old2 + flatten(kept)
    messages = [system, *history, current]
    return messages, system, kept, current


async def test_over_budget_summarizes_old_turns_keeps_last_four_turns_and_current():
    messages, system, kept, current = _build_over_budget_conversation()
    original = list(messages)

    summary_text = "Mirabelle Kovacs, ref 4817; decision: on garde PostgreSQL."
    llm = FakeLLM(script=[summary_text], max_context_tokens=500)
    state: dict = {}

    result = await apply_budget(messages, llm, state, reserve_output=0, trigger_ratio=0.01)

    # input untouched
    assert messages == original
    assert result is not messages

    # the LLM was called once, and the old turns were in the prompt
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"]
    assert prompt[0].role == "system"
    assert prompt[1].role == "user"
    assert "Mirabelle Kovacs" in prompt[1].content
    assert "4817" in prompt[1].content
    assert "on garde PostgreSQL" in prompt[1].content

    # summary injected into the system message, alongside the original content
    assert result[0].role == "system"
    assert system.content in result[0].content
    assert summary_text in result[0].content

    # last 4 turns + current message intact, in order
    tail = result[1:]
    expected_tail = [m.content for m in flatten(kept)] + [current.content]
    assert [m.content for m in tail] == expected_tail

    # stats
    stats = state["context_stats"]
    assert stats["compressed"] is True
    assert stats["tokens_before"] > stats["tokens_after"]

    # summary persisted, covering exactly the two old turns (4 messages)
    saved = state["summary"]
    assert saved["text"] == summary_text
    assert saved["covered"] == 4
    assert saved["digest"] == _digest(messages[1:5])


# --- reuse of a saved summary, no recompute -------------------------------------------


async def test_matching_saved_summary_is_reused_without_calling_the_llm():
    messages, system, kept, current = _build_over_budget_conversation()
    old_messages = messages[1:5]  # old1 + old2

    saved_text = "Previously summarized: Mirabelle Kovacs, ref 4817, on garde PostgreSQL."
    state = {
        "summary": {"text": saved_text, "covered": 4, "digest": _digest(old_messages)},
    }
    llm = FakeLLM(script=[], max_context_tokens=500)  # any chat() call would raise (empty script)

    result = await apply_budget(list(messages), llm, state, reserve_output=0, trigger_ratio=0.01)

    assert llm.calls == []
    assert saved_text in result[0].content
    assert state["summary"]["text"] == saved_text
    assert state["summary"]["covered"] == 4


# --- fork / other branch: mismatched digest is ignored, not deleted ------------------


async def test_forked_summary_digest_mismatch_is_ignored_and_overwritten():
    messages, system, kept, current = _build_over_budget_conversation()

    stale_text = "STALE SUMMARY FROM ANOTHER BRANCH"
    state = {
        "summary": {"text": stale_text, "covered": 4, "digest": "not-the-real-digest"},
    }
    fresh_text = "Fresh summary: Mirabelle Kovacs, ref 4817, on garde PostgreSQL."
    llm = FakeLLM(script=[fresh_text], max_context_tokens=500)

    result = await apply_budget(list(messages), llm, state, reserve_output=0, trigger_ratio=0.01)

    # the stale summary was not trusted: the LLM was called to recompute
    assert len(llm.calls) == 1
    assert stale_text not in result[0].content
    assert fresh_text in result[0].content

    # it is overwritten now that a new summary was actually computed
    assert state["summary"]["text"] == fresh_text
    assert state["summary"]["covered"] == 4
    assert state["summary"]["digest"] == _digest(messages[1:5])


# --- LLM failure: drop the oldest turns instead ---------------------------------------


async def test_llm_failure_falls_back_to_dropping_oldest_turns(caplog):
    messages, system, kept, current = _build_over_budget_conversation()
    original = list(messages)

    llm = FakeLLM(script=[], max_context_tokens=500)  # chat() raises (empty script)
    state: dict = {}

    with caplog.at_level("WARNING"):
        result = await apply_budget(
            list(messages), llm, state, reserve_output=0, trigger_ratio=0.01
        )

    assert messages == original  # still not mutated

    # the old turns are gone, the recent ones and the current message survive
    tail_contents = [m.content for m in result if m.role != "system"]
    assert "Mirabelle Kovacs" not in " ".join(tail_contents)
    expected_tail = [m.content for m in flatten(kept)] + [current.content]
    assert tail_contents == expected_tail

    assert any("dropping oldest turns" in r.getMessage() for r in caplog.records)

    stats = state["context_stats"]
    assert stats["compressed"] is True
    assert stats["tokens_before"] > stats["tokens_after"]

    # no summary was computed, so nothing overwrites state["summary"]
    assert "summary" not in state
