"""Unit tests for `OutputGuardrailNode` (EV3, wiki §7.7)."""

from __future__ import annotations

import logging

import pytest

from core.agent import Agent
from core.llm import ILLM, LLMError
from core.node import Next, PipelineNode
from core.types import Context
from nodes.output_guardrail import OutputGuardrailNode
from nodes.react_loop import ReActLoopNode
from tests.fakes import FakeLLM

EMAIL = "jean.dupont@example.com"
PHONE = "06 12 34 56 78"
IBAN = "FR76 3000 6000 0112 3456 7890 189"
CARD = "4111 1111 1111 1111"


class _Failing(ILLM):
    model = "failing"
    max_context_tokens = 8192

    def count_tokens(self, messages):
        return 0

    @property
    def options(self) -> dict:
        return {}

    def update_options(self, **kwargs) -> None:
        pass

    async def chat(self, messages, tools=None, schema=None):
        raise LLMError("down")
        yield  # pragma: no cover


class _Spy(PipelineNode):
    def __init__(self) -> None:
        self.called = False

    async def handle(self, ctx: Context, next: Next) -> Context:
        self.called = True
        return ctx


async def run(node: OutputGuardrailNode, response: str | None, llm=None) -> Context:
    ctx = Context("q", [], llm or FakeLLM([]), {}, response=response)

    async def terminal(c: Context) -> Context:
        return c

    return await node.handle(ctx, terminal)


@pytest.mark.parametrize(
    ("raw", "tag"),
    [(EMAIL, "EMAIL"), (PHONE, "PHONE"), (IBAN, "IBAN"), (CARD, "CARD")],
)
async def test_masks_each_pii_type(raw, tag):
    ctx = await run(OutputGuardrailNode(), f"Voici {raw} ok")
    assert ctx.response == f"Voici [{tag}] ok"
    assert ctx.state["pii_found"] == [{"type": tag.lower(), "count": 1}]
    assert raw not in repr(ctx.state)
    assert ctx.state["output_modified"] is True


async def test_pii_counts_and_no_pii_sets_empty_list():
    ctx = await run(OutputGuardrailNode(), f"{EMAIL} et autre@example.org")
    assert ctx.state["pii_found"] == [{"type": "email", "count": 2}]
    clean = await run(OutputGuardrailNode(), "rien à signaler")
    assert clean.state["pii_found"] == []
    assert clean.state["output_modified"] is False
    assert clean.state["output_warnings"] == []
    assert clean.response == "rien à signaler"


async def test_pii_refuse():
    node = OutputGuardrailNode(pii_action="refuse", refusal_message="non")
    ctx = await run(node, f"mail {EMAIL}")
    assert ctx.response == "non"
    assert ctx.state["output_modified"] is True
    assert EMAIL not in repr(ctx.state)


async def test_pii_warn_keeps_answer_and_logs_name_only(caplog):
    node = OutputGuardrailNode(pii_action="warn")
    with caplog.at_level(logging.WARNING):
        ctx = await run(node, f"mail {EMAIL}")
    assert ctx.response == f"mail {EMAIL}"
    assert ctx.state["output_warnings"] == ["pii"]
    assert ctx.state["output_modified"] is False
    assert EMAIL not in caplog.text
    assert "output guardrail: pii" in caplog.text


async def test_only_selected_pii_types():
    ctx = await run(OutputGuardrailNode(pii=["email"]), f"{EMAIL} {PHONE}")
    assert ctx.response == f"[EMAIL] {PHONE}"


async def test_forbidden_term_refuse_case_insensitive():
    node = OutputGuardrailNode(forbidden_terms=["secret"], refusal_message="refusé")
    ctx = await run(node, "Le SeCrEt est ici.")
    assert ctx.response == "refusé"
    assert ctx.state["output_modified"] is True


async def test_forbidden_term_warn():
    node = OutputGuardrailNode(forbidden_terms=["secret"], terms_action="warn")
    ctx = await run(node, "un secret")
    assert ctx.response == "un secret"
    assert ctx.state["output_warnings"] == ["term"]


async def test_forbidden_term_word_boundary():
    node = OutputGuardrailNode(forbidden_terms=["cat"])
    ctx = await run(node, "concatenate the category")
    assert ctx.response == "concatenate the category"
    assert (await run(node, "a cat sat")).response != "a cat sat"


async def test_forbidden_term_with_regex_chars_is_escaped():
    node = OutputGuardrailNode(forbidden_terms=["c++"])
    assert (await run(node, "j'aime c++ beaucoup")).response != "j'aime c++ beaucoup"
    assert (await run(node, "cccc")).response == "cccc"


def _verdict(safe: bool, category: str) -> FakeLLM:
    import json

    return FakeLLM([json.dumps({"safe": safe, "category": category})])


async def test_classifier_unsafe_refuses():
    llm = _verdict(False, "violence")
    ctx = await run(OutputGuardrailNode(classifier=True), "du texte", llm)
    assert ctx.response == "Je ne peux pas fournir cette réponse."
    assert llm.calls[0]["schema"] is not None


async def test_classifier_unsafe_warn():
    node = OutputGuardrailNode(classifier=True, classifier_action="warn")
    ctx = await run(node, "du texte", _verdict(False, "violence"))
    assert ctx.response == "du texte"
    assert ctx.state["output_warnings"] == ["classifier:violence"]


async def test_classifier_safe_unchanged():
    ctx = await run(OutputGuardrailNode(classifier=True), "du texte", _verdict(True, "ok"))
    assert ctx.response == "du texte"
    assert ctx.state["output_modified"] is False


async def test_classifier_llm_error_fails_open():
    ctx = await run(OutputGuardrailNode(classifier=True), "du texte", _Failing())
    assert ctx.response == "du texte"
    assert ctx.state["output_warnings"] == ["classifier_error"]


async def test_classifier_skipped_when_already_refused():
    llm = FakeLLM([])
    node = OutputGuardrailNode(classifier=True, forbidden_terms=["x"])
    ctx = await run(node, "x", llm)
    assert ctx.response == "Je ne peux pas fournir cette réponse."
    assert llm.calls == []


async def test_response_none_calls_next_and_writes_nothing():
    spy = _Spy()
    agent = Agent(FakeLLM([]), [], [OutputGuardrailNode(), spy])
    ctx = await agent.run("q", [])
    assert spy.called
    assert ctx.response is None
    assert ctx.state == {}


def test_invalid_actions_raise():
    with pytest.raises(ValueError):
        OutputGuardrailNode(pii_action="block")
    with pytest.raises(ValueError):
        OutputGuardrailNode(terms_action="mask")
    with pytest.raises(ValueError):
        OutputGuardrailNode(classifier_action="mask")


def test_from_config_rules_file():
    node = OutputGuardrailNode.from_config({"rules": "config/guardrails.yaml"}, object())
    assert node.pii_action == "mask"
    assert node.classifier is False
    assert node.refusal_message == "Je ne peux pas fournir cette réponse."


def test_from_config_explicit_params_override_rules():
    node = OutputGuardrailNode.from_config(
        {"rules": "config/guardrails.yaml", "pii_action": "refuse", "strict": True}, object()
    )
    assert node.pii_action == "refuse"
    assert node.strict is True


def test_from_config_rules_fallback_to_config_dir_parent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "g.yaml").write_text(
        "output:\n  forbidden_terms:\n    action: warn\n    terms: [foo]\n  classifier:\n"
        "    enabled: true\n    action: warn\n"
    )
    monkeypatch.chdir(tmp_path.parent)
    deps = SimpleNamespace(config_dir=tmp_path / "config")
    node = OutputGuardrailNode.from_config({"rules": "config/g.yaml"}, deps)
    assert node.terms_action == "warn"
    assert node.classifier is True
    assert node.classifier_action == "warn"


def test_from_config_unknown_param_and_bad_action_raise():
    with pytest.raises(ValueError, match="Unknown"):
        OutputGuardrailNode.from_config({"nope": 1}, object())
    with pytest.raises(ValueError):
        OutputGuardrailNode.from_config({"pii_action": "explode"}, object())


class _Recorder(PipelineNode):
    """Stands in for CacheNode: stores `ctx.response` on the way back."""

    def __init__(self) -> None:
        self.seen: str | None = None

    async def handle(self, ctx: Context, next: Next) -> Context:
        ctx = await next(ctx)
        self.seen = ctx.response
        return ctx


async def test_acceptance_earlier_node_sees_masked_answer():
    recorder = _Recorder()
    agent = Agent(
        FakeLLM([f"Écris-moi à {EMAIL}"]), [], [recorder, ReActLoopNode(), OutputGuardrailNode()]
    )
    ctx = await agent.run("q", [])
    assert recorder.seen is not None
    assert "[EMAIL]" in recorder.seen
    assert EMAIL not in recorder.seen
    assert ctx.response == recorder.seen


async def test_streaming_leaks_raw_tokens_but_flags_modification():
    tokens: list[str] = []
    agent = Agent(
        FakeLLM([f"Écris-moi à {EMAIL}"]),
        [],
        [ReActLoopNode(on_token=tokens.append), OutputGuardrailNode()],
    )
    ctx = await agent.run("q", [])
    assert "".join(tokens) == f"Écris-moi à {EMAIL}"
    assert ctx.state["output_modified"] is True
    assert EMAIL not in ctx.response
