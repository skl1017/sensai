"""Unit tests for `ReActLoopNode` (#14/#15/#16): the LLM <-> tools loop.

`FakeLLM`/`FakeTool` (from `tests.fakes`) cover most scenarios. A couple of
edge cases need more control than `FakeLLM` offers (a single turn carrying
both a "thought" and tool calls, or precise chunk boundaries for the
`on_token` test): those use a small locally-defined `ChunkLLM`, in the same
spirit as `StubLLM` in `test_agent.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from core.llm import ILLM
from core.types import Context, LLMChunk, Message, ToolCall
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeLLM, FakeTool


class ChunkLLM(ILLM):
    """Locally-defined `ILLM` double that replays raw `LLMChunk` sequences per call.

    Needed where `FakeLLM` can't help: producing a single turn that carries both
    text (a "thought") and tool calls together, or controlling exact chunk
    boundaries for the `on_token` concatenation test.
    """

    model = "chunk-llm"
    max_context_tokens = 8192

    def __init__(self, turns: list[list[LLMChunk]]) -> None:
        self.turns = list(turns)
        self.calls: list[dict] = []

    def count_tokens(self, messages: list[Message]) -> int:
        return 0

    async def chat(self, messages, tools=None, schema=None):
        self.calls.append({"messages": list(messages), "tools": tools, "schema": schema})
        for chunk in self.turns.pop(0):
            yield chunk


async def _next(ctx: Context) -> Context:
    return ctx


def make_next():
    seen: list[Context] = []

    async def next_(ctx: Context) -> Context:
        seen.append(ctx)
        return ctx

    return next_, seen


def calculator_tool() -> FakeTool:
    return FakeTool(
        name="calculator",
        result=lambda a, b: str(a + b),
        description="Adds two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


async def test_calculator_tool_call_then_final_answer():
    call = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2})
    llm = FakeLLM([[call], "The answer is 3"])
    tool = calculator_tool()
    ctx = Context("what is 1 + 2?", [], llm, {"calculator": tool})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    assert result.response == "The answer is 3"
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    assert result.messages[0] == Message("user", "what is 1 + 2?")
    assert result.messages[1].tool_calls == [call]
    assert result.messages[2] == Message("tool", "3", tool_call_id="1", name="calculator")
    assert result.messages[3] == Message("assistant", "The answer is 3")

    assert tool.calls == [{"a": 1, "b": 2}]

    trace = result.state["react_trace"]
    assert len(trace) == 1
    assert set(trace[0]) == {"step", "thought", "action", "args", "observation"}
    assert trace[0] == {
        "step": 1,
        "thought": "",
        "action": "calculator",
        "args": {"a": 1, "b": 2},
        "observation": "3",
    }
    assert result.state["react_truncated"] is False

    assert len(llm.calls) == 2
    assert llm.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    ]


async def test_thought_text_recorded_across_turns():
    call1 = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2})
    call2 = ToolCall(id="2", name="calculator", arguments={"a": 3, "b": 4})
    llm = ChunkLLM(
        [
            [LLMChunk(text="Let me add the first pair.", tool_calls=[call1]), LLMChunk(done=True)],
            [LLMChunk(text="Now the second pair.", tool_calls=[call2]), LLMChunk(done=True)],
            [LLMChunk(text="Done: 3 and 7."), LLMChunk(done=True)],
        ]
    )
    tool = calculator_tool()
    ctx = Context("add two pairs", [], llm, {"calculator": tool})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    trace = result.state["react_trace"]
    assert [t["step"] for t in trace] == [1, 2]
    assert trace[0]["thought"] == "Let me add the first pair."
    assert trace[1]["thought"] == "Now the second pair."
    assert result.response == "Done: 3 and 7."


async def test_multiple_calls_in_one_turn_run_sequentially():
    events: list[str] = []

    def make_slow(label: str, delay: float):
        async def _run(**kwargs):
            events.append(f"start:{label}")
            await asyncio.sleep(delay)
            events.append(f"end:{label}")
            return label

        return _run

    tool_a = FakeTool(name="a", result=make_slow("a", 0.03))
    tool_b = FakeTool(name="b", result=make_slow("b", 0.01))
    call_a = ToolCall(id="1", name="a", arguments={})
    call_b = ToolCall(id="2", name="b", arguments={})
    llm = FakeLLM([[call_a, call_b], "done"])
    ctx = Context("go", [], llm, {"a": tool_a, "b": tool_b})
    node = ReActLoopNode()

    await node.handle(ctx, _next)

    assert events == ["start:a", "end:a", "start:b", "end:b"]


async def test_unknown_tool_observation_lists_available_tools_and_continues():
    call = ToolCall(id="1", name="nonexistent", arguments={})
    llm = FakeLLM([[call], "ok final"])
    tool = calculator_tool()
    ctx = Context("go", [], llm, {"calculator": tool})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    obs = result.state["react_trace"][0]["observation"]
    assert obs == "Error: unknown tool 'nonexistent'. Available tools: calculator."
    assert result.response == "ok final"


async def test_tool_exception_becomes_observation_and_does_not_fail_request():
    call = ToolCall(id="1", name="boom_tool", arguments={})
    llm = FakeLLM([[call], "recovered"])
    tool = FakeTool(name="boom_tool", result=ValueError("boom"))
    ctx = Context("go", [], llm, {"boom_tool": tool})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    assert result.state["react_trace"][0]["observation"] == "Error: ValueError: boom"
    assert result.response == "recovered"


async def test_tool_timeout_becomes_observation():
    async def sleepy(**kwargs):
        await asyncio.sleep(0.2)
        return "too late"

    call = ToolCall(id="1", name="slow", arguments={})
    llm = FakeLLM([[call], "final"])
    tool = FakeTool(name="slow", result=sleepy)
    ctx = Context("go", [], llm, {"slow": tool})
    node = ReActLoopNode(tool_timeout=0.05)

    result = await node.handle(ctx, _next)

    obs = result.state["react_trace"][0]["observation"]
    assert obs == "Error: tool 'slow' timed out after 0.05s."


async def test_tool_enforcing_own_timeout_is_not_cut_off():
    async def sleepy(**kwargs):
        await asyncio.sleep(0.1)
        return "finished anyway"

    call = ToolCall(id="1", name="slow", arguments={})
    llm = FakeLLM([[call], "final"])
    tool = FakeTool(name="slow", result=sleepy, enforces_own_timeout=True)
    ctx = Context("go", [], llm, {"slow": tool})
    node = ReActLoopNode(tool_timeout=0.05)

    result = await node.handle(ctx, _next)

    assert result.state["react_trace"][0]["observation"] == "finished anyway"


async def test_budget_exhausted_makes_final_call_without_tools():
    call1 = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 1})
    call2 = ToolCall(id="2", name="calculator", arguments={"a": 2, "b": 2})
    llm = FakeLLM([[call1], [call2], "final text"])
    tool = calculator_tool()
    ctx = Context("go", [], llm, {"calculator": tool})
    node = ReActLoopNode(max_iterations=2)

    result = await node.handle(ctx, _next)

    assert result.state["react_truncated"] is True
    assert result.response == "final text"
    assert llm.calls[-1]["tools"] is None
    assert any(
        m.role == "system" and "iteration budget exhausted" in m.content for m in result.messages
    )


async def test_normal_completion_does_not_set_truncated():
    llm = FakeLLM(["just an answer"])
    ctx = Context("go", [], llm, {})
    node = ReActLoopNode(max_iterations=2)

    result = await node.handle(ctx, _next)

    assert result.state["react_truncated"] is False


async def test_observation_is_truncated_to_max_observation_chars():
    call = ToolCall(id="1", name="verbose", arguments={})
    llm = FakeLLM([[call], "final"])
    tool = FakeTool(name="verbose", result="x" * 50)
    ctx = Context("go", [], llm, {"verbose": tool})
    node = ReActLoopNode(max_observation_chars=10)

    result = await node.handle(ctx, _next)

    obs = result.state["react_trace"][0]["observation"]
    assert obs.startswith("x" * 10)
    assert "truncated, 40 chars omitted" in obs


async def test_repetition_detected_after_three_identical_calls():
    same = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2})
    llm = FakeLLM([[same], [same], [same], "final"])
    tool = calculator_tool()
    ctx = Context("go", [], llm, {"calculator": tool})
    node = ReActLoopNode(max_iterations=5)

    result = await node.handle(ctx, _next)

    assert len(tool.calls) == 2
    trace = result.state["react_trace"]
    assert trace[0]["observation"] == "3"
    assert trace[1]["observation"] == "3"
    assert trace[2]["observation"].startswith("Warning:")
    assert "3 times in a row" in trace[2]["observation"]
    assert result.response == "final"


async def test_repetition_counter_resets_on_different_arguments():
    call1 = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2})
    call2 = ToolCall(id="2", name="calculator", arguments={"a": 1, "b": 2})
    call3 = ToolCall(id="3", name="calculator", arguments={"a": 9, "b": 9})
    llm = FakeLLM([[call1], [call2], [call3], "final"])
    tool = calculator_tool()
    ctx = Context("go", [], llm, {"calculator": tool})
    node = ReActLoopNode(max_iterations=5)

    result = await node.handle(ctx, _next)

    assert len(tool.calls) == 3
    trace = result.state["react_trace"]
    assert not any(t["observation"].startswith("Warning:") for t in trace)


async def test_on_event_receives_thought_tool_call_observation_final_sync():
    events: list[tuple[str, dict]] = []

    def on_event(kind: str, data: dict) -> None:
        events.append((kind, dict(data)))

    call1 = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2})
    llm = ChunkLLM(
        [
            [LLMChunk(text="thinking...", tool_calls=[call1]), LLMChunk(done=True)],
            [LLMChunk(text="done."), LLMChunk(done=True)],
        ]
    )
    tool = calculator_tool()
    ctx = Context("go", [], llm, {"calculator": tool})
    node = ReActLoopNode(on_event=on_event)

    await node.handle(ctx, _next)

    kinds = [k for k, _ in events]
    assert kinds == ["thought", "tool_call", "observation", "final"]
    assert events[0][1] == {"step": 1, "text": "thinking..."}
    assert events[1][1] == {"step": 1, "name": "calculator", "args": {"a": 1, "b": 2}}
    assert events[2][1] == {"step": 1, "name": "calculator", "observation": "3"}
    assert events[3][1] == {"text": "done.", "truncated": False}


async def test_on_event_supports_async_callback():
    events: list[str] = []

    async def on_event(kind: str, data: dict) -> None:
        events.append(kind)

    llm = FakeLLM(["a plain final answer"])
    ctx = Context("go", [], llm, {})
    node = ReActLoopNode(on_event=on_event)

    await node.handle(ctx, _next)

    assert events == ["final"]


async def test_on_token_receives_streamed_fragments_concatenating_to_full_text():
    tokens: list[str] = []

    def on_token(text: str) -> None:
        tokens.append(text)

    llm = FakeLLM(["hello there this is a streamed answer"])
    ctx = Context("go", [], llm, {})
    node = ReActLoopNode(on_token=on_token)

    result = await node.handle(ctx, _next)

    assert "".join(tokens) == "hello there this is a streamed answer"
    assert result.response == "hello there this is a streamed answer"


async def test_cancelled_error_from_tool_propagates():
    call = ToolCall(id="1", name="cancels", arguments={})
    llm = FakeLLM([[call], "never reached"])
    tool = FakeTool(name="cancels", result=asyncio.CancelledError())
    ctx = Context("go", [], llm, {"cancels": tool})
    node = ReActLoopNode()

    with pytest.raises(asyncio.CancelledError):
        await node.handle(ctx, _next)


async def test_external_cancellation_while_tool_runs_propagates():
    started = asyncio.Event()

    async def slow(**kwargs):
        started.set()
        await asyncio.sleep(5)
        return "done"

    call = ToolCall(id="1", name="slow", arguments={})
    llm = FakeLLM([[call], "never reached"])
    tool = FakeTool(name="slow", result=slow)
    ctx = Context("go", [], llm, {"slow": tool})
    node = ReActLoopNode(tool_timeout=10)

    task = asyncio.create_task(node.handle(ctx, _next))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_next_called_exactly_once_with_response_set():
    llm = FakeLLM(["final answer"])
    ctx = Context("go", [], llm, {})
    node = ReActLoopNode()
    next_, seen = make_next()

    result = await node.handle(ctx, next_)

    assert len(seen) == 1
    assert seen[0] is result
    assert result.response == "final answer"


async def test_no_tools_passes_none_to_llm():
    llm = FakeLLM(["ok"])
    ctx = Context("go", [], llm, {})
    node = ReActLoopNode()

    await node.handle(ctx, _next)

    assert llm.calls[0]["tools"] is None


async def test_user_message_fallback_appended_when_missing():
    llm = FakeLLM(["ok"])
    ctx = Context("hello agent", [], llm, {})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    assert result.messages[0] == Message("user", "hello agent")
    assert sum(1 for m in result.messages if m.role == "user") == 1


async def test_user_message_fallback_not_duplicated_when_already_last():
    llm = FakeLLM(["ok"])
    existing = [Message("user", "hello agent")]
    ctx = Context("hello agent", existing, llm, {})
    node = ReActLoopNode()

    result = await node.handle(ctx, _next)

    assert sum(1 for m in result.messages if m.role == "user") == 1
    assert result.messages[0] is existing[0]


async def test_from_config_wires_ui_callbacks_and_passes_params():
    class DummyUI:
        def on_token(self, text: str) -> None:
            pass

        def on_event(self, kind: str, data: dict) -> None:
            pass

    class Deps:
        def __init__(self):
            self.ui = DummyUI()

    deps = Deps()
    node = ReActLoopNode.from_config({"max_iterations": 3, "tool_timeout": 5.0}, deps)

    assert node.max_iterations == 3
    assert node.tool_timeout == 5.0
    assert node.on_token == deps.ui.on_token
    assert node.on_event == deps.ui.on_event


async def test_from_config_without_ui_leaves_callbacks_none():
    class Deps:
        pass

    node = ReActLoopNode.from_config({"max_iterations": 4}, Deps())

    assert node.max_iterations == 4
    assert node.on_token is None
    assert node.on_event is None
