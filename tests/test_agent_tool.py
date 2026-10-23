"""Unit tests for `AgentTool` and `SubAgentUI` (tools/agent_tool.py), no network."""

from __future__ import annotations

import asyncio

import pytest

from core import runtime
from core.agent import Agent
from core.node import Next, PipelineNode
from core.types import Context, ToolCall
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeLLM, FakeTool, FakeUI
from tools.agent_tool import MAX_ANSWER_CHARS, TRUNCATED_NOTE, AgentTool, SubAgentUI
from tools.approved_tool import REFUSAL, ApprovedTool


class CapturingNode(PipelineNode):
    """Records the sub-agent's state and runtime depth, then answers."""

    def __init__(self, answer: str = "done", truncated: bool = False):
        self.answer = answer
        self.truncated = truncated
        self.seen: dict = {}

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.seen = {"state": dict(ctx.state), "depth": runtime.depth.get()}
        ctx.response = self.answer
        if self.truncated:
            ctx.state["react_truncated"] = True
        return ctx


class SlowNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        await asyncio.sleep(10)
        return ctx


class RaisingNode(PipelineNode):
    def __init__(self, exc: BaseException):
        self.exc = exc

    async def handle(self, ctx: Context, next: Next) -> Context:
        raise self.exc


class SilentNode(PipelineNode):
    async def handle(self, ctx: Context, next: Next) -> Context:
        return ctx


def sub_agent(*nodes: PipelineNode, llm=None, tools=None) -> Agent:
    return Agent(llm or FakeLLM([]), tools or [], list(nodes))


def react_sub_agent(script, tools=None, ui=None) -> tuple[Agent, FakeLLM]:
    llm = FakeLLM(script)
    callbacks = {"on_token": ui.on_token, "on_event": ui.on_event} if ui else {}
    return Agent(llm, tools or [], [ReActLoopNode(**callbacks)]), llm


def call(name: str, **arguments) -> list[ToolCall]:
    return [ToolCall(id="c1", name=name, arguments=arguments)]


# ---------- schema ----------


def test_name_description_and_schema():
    tool = AgentTool("researcher", "Searches the ingested documents.", sub_agent())

    assert tool.name == "ask_researcher"
    assert tool.description.startswith("Searches the ingested documents.")
    assert "does not see this conversation" in tool.description
    assert tool.parameters["required"] == ["task"]
    assert tool.parameters["additionalProperties"] is False
    assert tool.enforces_own_timeout is True


@pytest.mark.parametrize("name", ["Researcher", "ask-me", "1st", ""])
def test_rejects_non_snake_case_name(name):
    with pytest.raises(ValueError, match="snake_case"):
        AgentTool(name, "d", sub_agent())


# ---------- run ----------


async def test_returns_sub_agent_final_answer():
    agent, _ = react_sub_agent(["Paris est la capitale."])

    assert await AgentTool("geo", "d", agent).run("capitale de la France ?") == (
        "Paris est la capitale."
    )


async def test_sub_agent_sees_only_the_task():
    agent, llm = react_sub_agent(["ok"])

    await AgentTool("geo", "d", agent).run("tâche autonome")

    (messages,) = [c["messages"] for c in llm.calls]
    assert [(m.role, m.content) for m in messages] == [("user", "tâche autonome")]


async def test_state_carries_depth_and_parent_trace_id():
    node = CapturingNode()
    tool = AgentTool("geo", "d", sub_agent(node))
    trace_token = runtime.trace_id.set("trace-123")
    try:
        await tool.run("t")
    finally:
        runtime.trace_id.reset(trace_token)

    assert node.seen["state"] == {"depth": 1, "parent_trace_id": "trace-123"}
    assert node.seen["depth"] == 1
    assert runtime.depth.get() == 0


async def test_refuses_beyond_max_depth_without_running_sub_agent():
    node = CapturingNode()
    tool = AgentTool("geo", "d", sub_agent(node), max_depth=2)
    token = runtime.depth.set(2)
    try:
        with pytest.raises(RuntimeError, match="maximum sub-agent depth"):
            await tool.run("t")
    finally:
        runtime.depth.reset(token)

    assert node.seen == {}


async def test_allows_call_just_below_max_depth():
    tool = AgentTool("geo", "d", sub_agent(CapturingNode("ok")), max_depth=2)
    token = runtime.depth.set(1)
    try:
        assert await tool.run("t") == "ok"
    finally:
        runtime.depth.reset(token)


async def test_max_calls_counts_in_shared_request_list():
    tool = AgentTool("geo", "d", sub_agent(CapturingNode("ok")), max_calls=2)
    calls: list[str] = ["other"]
    token = runtime.subagent_calls.set(calls)
    try:
        await tool.run("1")
        await tool.run("2")
        with pytest.raises(RuntimeError, match="already called 2 times"):
            await tool.run("3")
    finally:
        runtime.subagent_calls.reset(token)

    assert calls == ["other", "geo", "geo"]


async def test_timeout_raises_explicit_error():
    tool = AgentTool("geo", "d", sub_agent(SlowNode()), timeout=0.05)

    with pytest.raises(RuntimeError, match="did not answer within 0.05s"):
        await tool.run("t")


async def test_timeout_error_from_inside_sub_agent_is_not_reported_as_own_timeout():
    tool = AgentTool("geo", "d", sub_agent(RaisingNode(TimeoutError("inner"))), timeout=5)

    with pytest.raises(TimeoutError, match="inner"):
        await tool.run("t")


async def test_sub_agent_error_propagates():
    tool = AgentTool("geo", "d", sub_agent(RaisingNode(ValueError("boom"))))

    with pytest.raises(ValueError, match="boom"):
        await tool.run("t")


async def test_no_answer_raises():
    tool = AgentTool("geo", "d", sub_agent(SilentNode()))

    with pytest.raises(RuntimeError, match="returned no answer"):
        await tool.run("t")


async def test_truncated_sub_agent_answer_gets_a_note():
    tool = AgentTool("geo", "d", sub_agent(CapturingNode("partiel", truncated=True)))

    assert await tool.run("t") == "partiel" + TRUNCATED_NOTE


async def test_long_answer_is_truncated_at_the_source():
    tool = AgentTool("geo", "d", sub_agent(CapturingNode("x" * (MAX_ANSWER_CHARS + 50))))

    assert len(await tool.run("t")) == MAX_ANSWER_CHARS


async def test_cancellation_propagates_and_resets_runtime():
    tool = AgentTool("geo", "d", sub_agent(SlowNode()))
    task = asyncio.create_task(tool.run("t"))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.depth.get() == 0


# ---------- through a parent ReAct loop ----------


async def test_parent_gets_sub_agent_answer_as_observation():
    sub, _ = react_sub_agent(["42"])
    parent_llm = FakeLLM([call("ask_math", task="6 * 7 ?"), "La réponse est 42."])
    parent = Agent(parent_llm, [AgentTool("math", "d", sub)], [ReActLoopNode()])

    ctx = await parent.run("combien font 6 * 7 ?", [])

    assert ctx.response == "La réponse est 42."
    observation = parent_llm.calls[1]["messages"][-1]
    assert observation.role == "tool" and observation.content == "42"


async def test_max_calls_is_enforced_per_root_request():
    sub, sub_llm = react_sub_agent(["un", "deux"])
    parent_llm = FakeLLM([call("ask_math", task="a"), call("ask_math", task="b"), "fin"])
    parent = Agent(parent_llm, [AgentTool("math", "d", sub, max_calls=1)], [ReActLoopNode()])

    await parent.run("q", [])

    assert len(sub_llm.calls) == 1
    assert "already called 1 times" in parent_llm.calls[2]["messages"][-1].content
    assert runtime.subagent_calls.get() is None


async def test_each_root_request_gets_a_fresh_call_budget():
    sub, _ = react_sub_agent(["un", "deux"])
    parent_llm = FakeLLM([call("ask_math", task="a"), "fin 1", call("ask_math", task="b"), "fin 2"])
    parent = Agent(parent_llm, [AgentTool("math", "d", sub, max_calls=1)], [ReActLoopNode()])

    await parent.run("q1", [])
    await parent.run("q2", [])

    assert parent_llm.calls[3]["messages"][-1].content == "deux"


async def test_nested_sub_agents_share_the_root_call_list():
    inner_node = CapturingNode("inner")
    seen: dict = {}

    class RecordingNode(PipelineNode):
        async def handle(self, ctx: Context, next: Next) -> Context:
            seen["calls"] = runtime.subagent_calls.get()
            return await next(ctx)

    inner_tool = AgentTool("inner", "d", sub_agent(RecordingNode(), inner_node))
    middle, _ = react_sub_agent([call("ask_inner", task="x"), "middle"], tools=[inner_tool])
    parent_llm = FakeLLM([call("ask_middle", task="y"), "fin"])
    parent = Agent(parent_llm, [AgentTool("middle", "d", middle)], [ReActLoopNode()])

    await parent.run("q", [])

    assert seen["calls"] == ["middle", "inner"]
    assert inner_node.seen["depth"] == 2


# ---------- SubAgentUI ----------


async def test_sub_agent_ui_prefixes_questions_for_the_parent_user():
    parent_ui = FakeUI(answers=["oui"])

    answer = await SubAgentUI(parent_ui, "researcher").ask("Continuer ?", kind="text")

    assert answer == "oui"
    assert parent_ui.questions == [("[researcher] Continuer ?", "text", None)]


def test_sub_agent_ui_mutes_tokens_and_tags_events():
    parent_ui = FakeUI()
    ui = SubAgentUI(parent_ui, "researcher")

    ui.on_token("secret")
    ui.on_event("tool_call", {"name": "calculator"})

    assert parent_ui.tokens == []
    assert parent_ui.events == [("tool_call", {"name": "calculator", "agent": "researcher"})]


async def test_nested_sub_agent_ui_builds_agent_path_and_prefixes():
    parent_ui = FakeUI()
    ui = SubAgentUI(SubAgentUI(parent_ui, "outer"), "inner")

    ui.on_event("final", {"text": "t"})
    await ui.ask("Q ?")

    assert parent_ui.events == [("final", {"text": "t", "agent": "outer/inner"})]
    assert parent_ui.questions[0][0] == "[outer] [inner] Q ?"


async def test_sub_agent_approval_reaches_parent_user_who_can_refuse():
    parent_ui = FakeUI(answers=["no"])
    sub_ui = SubAgentUI(parent_ui, "writer")
    write = FakeTool("write_file", result="written")
    sub, sub_llm = react_sub_agent(
        [call("write_file"), "je n'ai pas pu écrire"],
        tools=[ApprovedTool(write, sub_ui)],
        ui=sub_ui,
    )

    answer = await AgentTool("writer", "d", sub).run("écris un fichier")

    assert parent_ui.questions[0][0].startswith("[writer] ")
    assert write.calls == []
    assert sub_llm.calls[1]["messages"][-1].content == REFUSAL
    assert answer == "je n'ai pas pu écrire"
    assert parent_ui.tokens == []
    assert ("final", {"text": "je n'ai pas pu écrire", "truncated": False, "agent": "writer"}) in (
        parent_ui.events
    )
