"""Serveur HTTP de l'agent : ASGI pur (sans framework), lancé par uvicorn.

    GET  /health
    POST /chat   {"text": "...", "session_id"?, "persona"?, "parent_id"?, "stream"?}

Avec "stream": true, la réponse est un flux SSE (events `token`, puis `done` ou `error`).
Sinon, un seul JSON. Fermer la connexion en cours de flux annule le tour (X2).
"""
import asyncio
import contextvars
import json
import os
import tempfile
import uuid

import yaml

from app import Stores, build_agent, chat_turn

CONFIG_PATH = os.environ.get("SENSAI_CONFIG", "config/agent.yaml")

_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar("queue", default=None)
_state: dict = {}


class ServerUI:
    """UserInterface sans utilisateur interactif : les tokens vont dans la file du tour en cours."""

    async def on_token(self, text: str) -> None:
        if (queue := _queue.get()) is not None:
            queue.put_nowait(("token", {"text": text}))

    async def ask(self, question: str, kind: str = "text", options=None) -> str:
        return "aucun utilisateur"


def _load_config() -> tuple[dict, str]:
    """Charge la config et surcharge l'hôte Ollama via OLLAMA_HOST (localhost ne marche pas en conteneur)."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if host := os.environ.get("OLLAMA_HOST"):
        cfg["llm"]["host"] = host
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg, path


async def _lifespan(receive, send):
    while True:
        event = await receive()
        if event["type"] == "lifespan.startup":
            path = None
            try:
                cfg, path = _load_config()
                _state["stores"] = Stores.open(cfg)
                _state["agent"] = await build_agent(path, ServerUI())
            except Exception as e:
                await send({"type": "lifespan.startup.failed", "message": f"{type(e).__name__}: {e}"})
                return
            finally:
                if path:
                    os.unlink(path)
            await send({"type": "lifespan.startup.complete"})
        elif event["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        event = await receive()
        body += event.get("body", b"")
        if not event.get("more_body"):
            return body


async def _start(send, status: int, content_type: str):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type.encode()), (b"cache-control", b"no-cache")]})


async def _json(send, status: int, payload: dict):
    await _start(send, status, "application/json")
    await send({"type": "http.response.body", "body": json.dumps(payload, default=str).encode()})


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def _result(ctx, session_id: str) -> dict:
    keys = ("blocked", "cache_hit", "metrics", "react_trace", "eval_scores")
    return {"session_id": session_id, "response": ctx.response,
            **{k: ctx.state[k] for k in keys if k in ctx.state}}


async def _chat(receive, send):
    try:
        req = json.loads(await _read_body(receive) or b"{}")
        text = req["text"]
        if not isinstance(text, str):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return await _json(send, 400, {"error": 'corps JSON attendu : {"text": "..."}'})

    session_id = req.get("session_id") or uuid.uuid4().hex
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        _queue.set(queue)
        try:
            ctx = await chat_turn(_state["agent"], _state["stores"], session_id, text,
                                  req.get("persona", "default"), req.get("parent_id"))
            queue.put_nowait(("done", _result(ctx, session_id)))
        except Exception as e:
            queue.put_nowait(("error", {"error": f"{type(e).__name__}: {e}"}))

    task = asyncio.create_task(run())
    stream = bool(req.get("stream"))
    try:
        if stream:
            await _start(send, 200, "text/event-stream")
        while True:
            event, data = await queue.get()
            if stream:
                await send({"type": "http.response.body", "body": _sse(event, data), "more_body": event == "token"})
            if event == "token":
                continue
            if not stream:
                await _json(send, 500 if event == "error" else 200, data)
            return
    finally:
        if not task.done():
            task.cancel()


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        return await _lifespan(receive, send)
    if scope["type"] != "http":
        return
    route = (scope["method"], scope["path"])
    if route == ("GET", "/health"):
        return await _json(send, 200, {"status": "ok"})
    if route == ("POST", "/chat"):
        return await _chat(receive, send)
    await _json(send, 404, {"error": "not found"})
