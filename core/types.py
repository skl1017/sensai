"""Core dataclasses shared by the whole pipeline.

Four dataclasses are enough: `Message` and `ToolCall` describe the
conversation, `LLMChunk` a streamed fragment of a response, and `Context`
the state of a single request as it flows through the pipeline.

Context invariants
-------------------
- `messages` is a shared mutable list: nodes append messages to it, they
  never remove any (except `ContextBuilderNode`, which rebuilds and
  compresses it).
- `response` stays `None` as long as no final answer exists; a node that
  short-circuits the chain (does not call `next`) must set it before
  returning.
- `state` is free space for nodes: each key has a single owner (the node
  that writes it), other nodes only read it, and the entry point may seed
  an initial value.

Reserved `ctx.state` keys
--------------------------
- `session_id`: written by the entry point (`chat_turn`); read by
  `PersistNode`, `LoggingNode`, memory tools (via `ContextVar`). Session
  identifier.
- `persona`: written by the entry point; read by `PersonaSection`. Name of
  the active persona.
- `parent_id`: written by the entry point; read by `PersistNode`. Node
  under which the exchange is appended (X2).
- `trace_id`: written by `LoggingNode`; read by `EvalNode` (`background`
  mode), `AgentTool` (via `ContextVar`). Request identifier.
- `blocked`: written by `InputGuardrailNode`; read by `PersistNode`,
  `LoggingNode`, `EvalNode`, `ArtifactNode`. Reason for the block, absent
  otherwise.
- `cache_hit`: written by `CacheNode`; read by `LoggingNode`, `EvalNode`,
  `ArtifactNode`. `True` if the response comes from the cache.
- `summary`: written by the entry point (loaded value), then by
  `ContextBuilderNode`; read by `ContextBuilderNode`, `PersistNode`.
  Rolling M2 summary and id of the last covered node.
- `context_stats`: written by `ContextBuilderNode`; read by `LoggingNode`.
  Tokens before and after compression.
- `retrieved_chunks`: written by `ContextBuilderNode`; read by `EvalNode`.
  Injected RAG chunks, with score and source.
- `react_trace`: written by `ReActLoopNode`; read by `LoggingNode`,
  `EvalNode`, `CacheNode`. List of `{step, thought, action, args,
  observation}`.
- `react_truncated`: written by `ReActLoopNode`; read by `LoggingNode`,
  `CacheNode`, `ArtifactNode`. `True` if the iteration budget was
  exhausted.
- `pii_found`: written by `OutputGuardrailNode`; read by `LoggingNode`.
  PII types detected in the output.
- `eval_scores`: written by `EvalNode`; read by `LoggingNode`, UI. Scores
  and unverifiable claims.
- `artifact`: written by the entry point (loaded value), then by
  `ArtifactNode`; read by `ArtifactSection`, `PersistNode`, X3 export.
  Current content of the living document.
- `metrics`: written by `LoggingNode`; read by tests, dashboard, UI.
  Latency, tokens, tool calls.
- `depth`, `parent_trace_id`: written by `AgentTool` (sub-agent initial
  state); read by `AgentTool`, `LoggingNode`. Nesting depth and parent
  request (A2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.llm import ILLM
    from core.tool import ITool


@dataclass
class ToolCall:
    id: str  # Ollama doesn't provide one: the adapter generates a uuid
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # role=assistant
    tool_call_id: str | None = None  # role=tool
    name: str | None = None  # role=tool


@dataclass
class LLMChunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    prompt_tokens: int | None = None  # set on the last chunk (EV3)
    completion_tokens: int | None = None


@dataclass
class Context:
    user_input: str
    messages: list[Message]
    llm: ILLM
    tools: dict[str, ITool]
    response: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
