"""`IReranker` adapter for a local cross-encoder, run by `fastembed` (ONNX, CPU).

A cross-encoder reads the query and one passage together and outputs a
single relevance logit, which is much more precise than the cosine between
two independently computed embeddings, but nothing can be precomputed: it is
only used as a second stage, on the candidates the vector search returned.

`fastembed` is an optional dependency (`pip install -e '.[rerank]'`): it is
imported, and the model downloaded into `cache_dir` and loaded, on the first
`rerank` call, so building the adapter stays cheap. Inference is synchronous
and CPU-bound, so it runs in `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

from core.llm import IReranker

DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


class FastEmbedReranker(IReranker):
    """Cross-encoder reranker; scores are the model's logits through a sigmoid.

    `encoder`, when given, replaces the `fastembed.TextCrossEncoder` built
    from `model` (tests inject a fake): any object with a
    `rerank(query, documents) -> Iterable[float]` method returning logits.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | None = "data/models",
        encoder: Any = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._encoder = encoder
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._encoder is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "fastembed is not installed: pip install -e '.[rerank]'"
                    ) from exc
                self._encoder = TextCrossEncoder(self.model, cache_dir=self.cache_dir)
            return self._encoder

    def _rerank_sync(self, query: str, passages: list[str]) -> list[float]:
        logits = self._load().rerank(query, passages)
        return [_sigmoid(float(logit)) for logit in logits]

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = await asyncio.to_thread(self._rerank_sync, query, passages)
        if len(scores) != len(passages):
            raise RuntimeError(
                f"{self.model} returned {len(scores)} scores for {len(passages)} passages"
            )
        return scores
