"""`ReActLoopNode`: the LLM <-> tools loop (thought, action, observation).

Alternates model calls and tool executions until the model answers without
requesting a tool, or the iteration budget runs out. The node knows nothing
about concrete tools: it only ever goes through `ctx.tools` (a
`dict[str, ITool]`) and the `ITool` interface, and never imposes a `schema`
on the model (that is `chat_json`'s job, not this node's).

Flow (per iteration/"step")
----------------------------
1. Call the LLM with the current `ctx.messages` and the tool specs built
   from `ctx.tools` (`tools=None` when there are no tools registered).
2. No tool calls in the response -> that is the final answer: set
   `ctx.response`, append the assistant message, emit a `"final"` event and
   call `next(ctx)`.
3. Tool calls -> the text produced before them is the "thought"; append the
   assistant message (with `tool_calls`), then run each call *sequentially*
   (so approval prompts stay readable) and append one `Message(role="tool")`
   observation per call. Every turn appends one entry per tool call to
   `ctx.state["react_trace"]` (read by `LoggingNode`/`EvalNode`):
   `{step, thought, action, args, observation}`.
4. Iterations left -> loop; budget exhausted ->
   `ctx.state["react_truncated"] = True`, a system instruction is appended
   telling the model to answer now with what it already knows, and one last
   call is made with `tools=None` (any tool call in that response is
   ignored). The node then calls `next(ctx)` exactly like the normal path.

Error handling
--------------
A tool error never fails the request: an unknown tool name, a timeout, or
any exception raised by `ITool.run` all become an observation the model can
read and react to (see `_run_tool`). `asyncio.CancelledError` (interruption,
X2) is never caught anywhere in this node and always propagates.

Fallback: `ContextBuilderNode` normally appends the user turn to
`ctx.messages` before this node runs. If that didn't happen (e.g. this node
is used on its own, or in a test), `handle` appends
`Message("user", ctx.user_input)` itself, so the loop still has something to
send to the model. It only does this when the last message isn't already
that same user turn, to avoid duplicating it.

Repetition guard: if the same tool is called with the exact same arguments
three times in a row (`REPEAT_LIMIT`), the third (and any further identical)
call is not executed; the observation is a warning telling the model to
reuse the previous observation, change its arguments, or answer.

Observability: `on_token(text)` is called with every non-empty streamed text
fragment; `on_event(kind, data)` is called for `"thought"`, `"tool_call"`,
`"observation"` and `"final"` (see `core/ui.py` for the payload shapes).
Both callbacks may be sync or async and default to no-ops.
"""

from __future__ import annotations

import asyncio
import inspect
import json

from core.node import Next, PipelineNode
from core.tool import ITool
from core.types import Context, Message, ToolCall
from core.ui import OnEvent, OnToken


async def _emit(cb, *args) -> None:
    """Call a sync-or-async callback, awaiting it if needed. No-op if `cb` is None."""
    if cb is None:
        return
    result = cb(*args)
    if inspect.isawaitable(result):
        await result


class ReActLoopNode(PipelineNode):
    REPEAT_LIMIT = 3

    def __init__(
        self,
        max_iterations: int = 8,
        tool_timeout: float = 30.0,
        max_observation_chars: int = 4000,
        on_token: OnToken | None = None,
        on_event: OnEvent | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.max_observation_chars = max_observation_chars
        self.on_token = on_token
        self.on_event = on_event

    @classmethod
    def from_config(cls, params: dict, deps) -> ReActLoopNode:
        """Build from `agent.yaml` params; wires callbacks from `deps.ui` when present.

        `deps` is intentionally untyped here (not `Deps`) so this module never has to
        import `app.py`, which would break the `nodes/` -> `core/` dependency rule.
        """
        ui = getattr(deps, "ui", None)
        extra = {}
        if ui is not None:
            extra = {"on_token": ui.on_token, "on_event": ui.on_event}
        return cls(**params, **extra)

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx.state["react_trace"] = []
        ctx.state["react_truncated"] = False

        last = ctx.messages[-1] if ctx.messages else None
        if not (last is not None and last.role == "user" and last.content == ctx.user_input):
            ctx.messages.append(Message("user", ctx.user_input))

        specs = self._tool_specs(ctx.tools)

        last_call_key: tuple[str, str] | None = None
        repeat_count = 0

        for step in range(1, self.max_iterations + 1):
            text, calls = await self._call_llm(ctx, specs)

            if not calls:
                ctx.response = text
                ctx.messages.append(Message("assistant", text))
                await _emit(self.on_event, "final", {"text": text, "truncated": False})
                return await next(ctx)

            ctx.messages.append(Message("assistant", text, tool_calls=calls))
            if text.strip():
                await _emit(self.on_event, "thought", {"step": step, "text": text})

            for call in calls:
                await _emit(
                    self.on_event,
                    "tool_call",
                    {"step": step, "name": call.name, "args": call.arguments},
                )

                key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                if key == last_call_key:
                    repeat_count += 1
                else:
                    last_call_key = key
                    repeat_count = 1

                if repeat_count >= self.REPEAT_LIMIT:
                    obs = (
                        f"Warning: `{call.name}` was called {repeat_count} times in a row "
                        "with the same arguments; the result will not change. Use the "
                        "previous observation, change the arguments, or give your final "
                        "answer."
                    )
                else:
                    obs = await self._run_tool(call, ctx.tools)

                ctx.messages.append(Message("tool", obs, tool_call_id=call.id, name=call.name))
                ctx.state["react_trace"].append(
                    {
                        "step": step,
                        "thought": text,
                        "action": call.name,
                        "args": call.arguments,
                        "observation": obs,
                    }
                )
                await _emit(
                    self.on_event,
                    "observation",
                    {"step": step, "name": call.name, "observation": obs},
                )

        ctx.state["react_truncated"] = True
        ctx.messages.append(
            Message(
                "system",
                "<instruction: iteration budget exhausted, no more tools, answer now "
                "with what you already know>",
            )
        )
        text, _calls = await self._call_llm(ctx, None)
        ctx.response = text.strip() or "No final answer could be produced within the budget."
        ctx.messages.append(Message("assistant", ctx.response))
        await _emit(self.on_event, "final", {"text": ctx.response, "truncated": True})
        return await next(ctx)

    async def _call_llm(self, ctx: Context, specs: list[dict] | None) -> tuple[str, list[ToolCall]]:
        text = ""
        calls: list[ToolCall] = []
        async for chunk in ctx.llm.chat(ctx.messages, tools=specs):
            if chunk.text:
                text += chunk.text
                await _emit(self.on_token, chunk.text)
            calls.extend(chunk.tool_calls)
        return text, calls

    async def _run_tool(self, call: ToolCall, tools: dict[str, ITool]) -> str:
        tool = tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(tools)) or "none"
            return self._truncate(
                f"Error: unknown tool '{call.name}'. Available tools: {available}."
            )

        try:
            coro = tool.run(**call.arguments)
            if tool.enforces_own_timeout:
                result = await coro
            else:
                result = await asyncio.wait_for(coro, self.tool_timeout)
        except TimeoutError:
            return f"Error: tool '{call.name}' timed out after {self.tool_timeout}s."
        except Exception as exc:  # a tool error is an observation, not a crash
            return self._truncate(f"Error: {type(exc).__name__}: {exc}")

        if not isinstance(result, str):
            result = str(result)
        return self._truncate(result)

    def _truncate(self, text: str) -> str:
        """Truncate `text` to `max_observation_chars`, then append a length marker.

        The marker is appended *after* truncating the body, so the final string is a
        few characters longer than `max_observation_chars` when truncation happens.
        """
        if len(text) <= self.max_observation_chars:
            return text
        omitted = len(text) - self.max_observation_chars
        return text[: self.max_observation_chars] + f"\n…[truncated, {omitted} chars omitted]"

    @staticmethod
    def _tool_specs(tools: dict[str, ITool]) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools.values()
        ]
