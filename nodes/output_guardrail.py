"""`OutputGuardrailNode`: inspects the final answer, masks PII or refuses (EV3, wiki §7.7).

Behavior
--------
Once a node downstream (normally `ReActLoopNode`) has produced
`ctx.response`, this node checks it, in this order:

1. Forbidden terms (case-insensitive, whole words): `terms_action` is
   `refuse` (the answer is replaced by `refusal_message`) or `warn`.
2. Optional LLM classifier (`classifier: true`): one schema-constrained
   `chat_json` call asks whether the answer is safe. Unsafe -> `classifier_action`
   (`refuse`/`warn`). An `LLMError` fails open (answer kept, warning
   `classifier_error`): a broken judge must not take the agent down.
3. PII (`core.pii.PIIDetector`) on the answer as it stands: `pii_action` is
   `mask` (`[EMAIL]`...), `refuse` or `warn`.

If `ctx.response` is `None` the node is transparent. Warnings are logged by
name only; matched values are never logged nor stored.

State keys written (this node is their only owner)
--------------------------------------------------
- `pii_found`: list of `{type, count}`, always set (`[]` when nothing found).
- `output_warnings`: list of rule names that fired in `warn` mode (or
  `classifier_error`), always set.
- `output_modified`: `True` when the answer differs from what the model wrote.

Placement
---------
After `ReActLoopNode` in the list, before `EvalNode`. `handle` does its work
*before* calling `next`, so everything placed earlier in the list (notably
`CacheNode` and `PersistNode`, which store on the way back) sees the already
masked answer: the cache never holds PII.

Streaming problem
-----------------
Tokens reach the UI while the model generates, before this node can look at
the full answer. Two strategies are supported:
- Strict (`strict: true`): the app wiring builds `ReActLoopNode` without
  `on_token` (`deps.stream = False`); nothing is shown until the checked
  answer is printed. No leak, no live streaming.
- Replacement (default): tokens are streamed as usual; afterwards the UI
  reprints `ctx.response` when `state["output_modified"]` is true. A brief
  leak of the raw text on screen is possible.
A sliding-window filter (mask on the fly) is NOT implemented: it is complex
and produces false positives at window boundaries. `strict` is only stored
here; the app wiring consumes it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from core.llm import LLMError, chat_json
from core.node import Next, PipelineNode
from core.pii import PII_TYPES, PIIDetector
from core.types import Context, Message

logger = logging.getLogger(__name__)

_KNOWN_PARAMS = {
    "strict",
    "pii",
    "pii_action",
    "forbidden_terms",
    "terms_action",
    "classifier",
    "refusal_message",
    "rules",
}
_PII_ACTIONS = ("mask", "refuse", "warn")
_REFUSE_WARN = ("refuse", "warn")
_DEFAULT_REFUSAL = "Je ne peux pas fournir cette réponse."

_CLASSIFIER_SYSTEM = (
    "Tu es un modérateur. On te donne la réponse d'un assistant. Indique si elle est "
    "sûre et appropriée (pas de contenu dangereux, illégal, haineux ou inapproprié). "
    "Réponds en JSON : `safe` (booléen) et `category` (courte étiquette, ex. "
    '"ok", "violence", "illegal").'
)
_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"safe": {"type": "boolean"}, "category": {"type": "string"}},
    "required": ["safe", "category"],
    "additionalProperties": False,
}


def _check(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")


def _load_rules(path: str, deps) -> dict:
    candidate = Path(path)
    if not candidate.exists():
        candidate = Path(getattr(deps, "config_dir", "config")).parent / path
    with open(candidate, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    section = data.get("output") or {}
    if not isinstance(section, dict):
        raise ValueError(f"{path}: `output` must be a mapping")
    return section


class OutputGuardrailNode(PipelineNode):
    def __init__(
        self,
        strict: bool = False,
        pii=PII_TYPES,
        pii_action: str = "mask",
        forbidden_terms=(),
        terms_action: str = "refuse",
        classifier: bool = False,
        classifier_action: str = "refuse",
        refusal_message: str = _DEFAULT_REFUSAL,
    ) -> None:
        _check(pii_action, _PII_ACTIONS, "pii_action")
        _check(terms_action, _REFUSE_WARN, "terms_action")
        _check(classifier_action, _REFUSE_WARN, "classifier_action")
        self.strict = strict
        self.pii_action = pii_action
        self.terms_action = terms_action
        self.classifier = classifier
        self.classifier_action = classifier_action
        self.refusal_message = refusal_message
        self._detector = PIIDetector(pii)
        terms = [t for t in forbidden_terms if t]
        self._terms_re = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
                re.IGNORECASE,
            )
            if terms
            else None
        )

    @classmethod
    def from_config(cls, params: dict, deps) -> OutputGuardrailNode:
        """Build from `agent.yaml` params, optionally seeded by a `rules:` YAML file.

        Explicit params override the file's `output:` section. Unknown params
        raise `ValueError`. `deps` is untyped so this module never imports `app.py`.
        """
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"Unknown OutputGuardrailNode param(s): {', '.join(sorted(unknown))}")

        kwargs: dict = {}
        if params.get("rules"):
            section = _load_rules(params["rules"], deps)
            if "refusal_message" in section:
                kwargs["refusal_message"] = section["refusal_message"]
            if "pii_action" in section:
                kwargs["pii_action"] = section["pii_action"]
            terms = section.get("forbidden_terms") or {}
            if "action" in terms:
                kwargs["terms_action"] = terms["action"]
            if terms.get("terms"):
                kwargs["forbidden_terms"] = terms["terms"]
            clf = section.get("classifier") or {}
            if "enabled" in clf:
                kwargs["classifier"] = bool(clf["enabled"])
            if "action" in clf:
                kwargs["classifier_action"] = clf["action"]

        for key in _KNOWN_PARAMS - {"rules"}:
            if key in params:
                kwargs[key] = params[key]
        return cls(**kwargs)

    async def handle(self, ctx: Context, next: Next) -> Context:
        if ctx.response is None:
            return await next(ctx)

        original = ctx.response
        warnings: list[str] = []
        refused = False

        if self._terms_re is not None and self._terms_re.search(ctx.response):
            if self.terms_action == "refuse":
                ctx.response = self.refusal_message
                refused = True
            else:
                warnings.append("term")

        if self.classifier and not refused:
            try:
                verdict = await chat_json(
                    ctx.llm,
                    [
                        Message("system", _CLASSIFIER_SYSTEM),
                        Message("user", ctx.response),
                    ],
                    schema=_CLASSIFIER_SCHEMA,
                )
                safe = bool(verdict.get("safe", True))
                category = str(verdict.get("category", "unknown"))
            except LLMError:
                warnings.append("classifier_error")
            else:
                if not safe:
                    if self.classifier_action == "refuse":
                        ctx.response = self.refusal_message
                        refused = True
                    else:
                        warnings.append(f"classifier:{category}")

        masked, found = self._detector.mask(ctx.response)
        ctx.state["pii_found"] = found
        if found:
            if self.pii_action == "mask":
                ctx.response = masked
            elif self.pii_action == "refuse":
                ctx.response = self.refusal_message
            else:
                warnings.append("pii")

        for name in warnings:
            logger.warning("output guardrail: %s", name)
        ctx.state["output_warnings"] = warnings
        ctx.state["output_modified"] = ctx.response != original
        return await next(ctx)
