from types import SimpleNamespace

import pytest

from nodes.rag_section import RAGSection

DIM = 4


class FakeEmbedder:
    """Embedder déterministe qui mémorise ce qu'on lui demande d'embedder."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] * DIM for t in texts]


class FakeChunkStore:
    """ChunkStore minimal : renvoie des hits préparés et mémorise les appels."""

    def __init__(self, hits=None):
        self.hits = hits or []
        self.calls: list[tuple[list[float], int]] = []

    async def search(self, embedding, k):
        self.calls.append((embedding, k))
        return self.hits


def make_chunk(
    chunk_id: str,
    text: str | None = None,
    source_id: str = "src1",
    source_name: str | None = None,
):
    return SimpleNamespace(
        id=chunk_id,
        text=text or f"texte {chunk_id}",
        source_id=source_id,
        source_name=source_name,
    )


def make_ctx(user_input: str = "combien de jours de congés ?", **state):
    return SimpleNamespace(user_input=user_input, state=dict(state))


@pytest.fixture
def embedder():
    return FakeEmbedder()


def make_section(embedder, hits=None, **kwargs):
    store = FakeChunkStore(hits)
    return RAGSection(embedder, store, **kwargs), store


# ---------- build ----------


@pytest.mark.asyncio
async def test_build_returns_none_when_no_hits(embedder):
    section, _ = make_section(embedder, hits=[])

    assert await section.build(make_ctx()) is None


@pytest.mark.asyncio
async def test_build_renders_numbered_passages_in_order(embedder):
    hits = [
        (make_chunk("c1", "Cinq semaines de congés."), 0.9),
        (make_chunk("c2", "RTT en plus."), 0.7),
    ]
    section, _ = make_section(embedder, hits=hits)

    text = await section.build(make_ctx())

    assert text.startswith("Relevant context")
    assert "[1] Cinq semaines de congés." in text
    assert "[2] RTT en plus." in text
    assert text.index("[1]") < text.index("[2]")


@pytest.mark.asyncio
async def test_build_filters_below_min_score(embedder):
    hits = [(make_chunk("good", "pertinent"), 0.8), (make_chunk("bad", "hors sujet"), 0.2)]
    section, _ = make_section(embedder, hits=hits, min_score=0.5)

    text = await section.build(make_ctx())

    assert "pertinent" in text
    assert "hors sujet" not in text


@pytest.mark.asyncio
async def test_build_returns_none_when_all_hits_below_min_score(embedder):
    section, _ = make_section(embedder, hits=[(make_chunk("c1"), 0.1)], min_score=0.5)
    ctx = make_ctx()

    assert await section.build(ctx) is None
    assert "retrieved_chunks" not in ctx.state


@pytest.mark.asyncio
async def test_build_keeps_hit_exactly_at_min_score(embedder):
    section, _ = make_section(embedder, hits=[(make_chunk("c1", "limite"), 0.5)], min_score=0.5)

    assert "limite" in await section.build(make_ctx())


@pytest.mark.asyncio
async def test_build_embeds_user_input_by_default(embedder):
    section, _ = make_section(embedder)

    await section.build(make_ctx("ma question"))

    assert embedder.calls == [["ma question"]]


@pytest.mark.asyncio
async def test_build_prefers_rewritten_query_from_state(embedder):
    section, _ = make_section(embedder)

    await section.build(make_ctx("et pour les cadres ?", query="congés payés des cadres"))

    assert embedder.calls == [["congés payés des cadres"]]


@pytest.mark.asyncio
async def test_build_passes_query_vector_and_k_to_store(embedder):
    section, store = make_section(embedder, k=7)

    await section.build(make_ctx("abc"))

    (vector, k) = store.calls[0]
    assert vector == [3.0] * DIM  # len("abc") == 3
    assert k == 7


@pytest.mark.asyncio
async def test_build_records_retrieved_chunks_in_state(embedder):
    hits = [
        (make_chunk("c1", source_id="A", source_name="rh.pdf"), 0.9),
        (make_chunk("c2", source_id="B"), 0.6),
    ]
    section, _ = make_section(embedder, hits=hits)
    ctx = make_ctx()

    await section.build(ctx)

    assert ctx.state["retrieved_chunks"] == [
        {"chunk_id": "c1", "source_id": "A", "source_name": "rh.pdf", "score": 0.9},
        {"chunk_id": "c2", "source_id": "B", "source_name": None, "score": 0.6},
    ]


@pytest.mark.asyncio
async def test_build_renders_source_name_under_its_passage(embedder):
    hits = [
        (make_chunk("c1", "Cinq semaines de congés.", source_name="rh.pdf"), 0.9),
        (make_chunk("c2", "RTT en plus.", source_name="accord.docx"), 0.7),
    ]
    section, _ = make_section(embedder, hits=hits)

    text = await section.build(make_ctx())

    assert "[1] Cinq semaines de congés.\n(source: rh.pdf)" in text
    assert "[2] RTT en plus.\n(source: accord.docx)" in text


@pytest.mark.asyncio
async def test_build_omits_source_line_when_source_name_unknown(embedder):
    section, _ = make_section(embedder, hits=[(make_chunk("c1", "Un texte."), 0.9)])

    text = await section.build(make_ctx())

    assert "[1] Un texte." in text
    assert "(source:" not in text


@pytest.mark.asyncio
async def test_build_records_only_chunks_kept_after_min_score_filter(embedder):
    hits = [(make_chunk("good"), 0.8), (make_chunk("bad"), 0.2)]
    section, _ = make_section(embedder, hits=hits, min_score=0.5)
    ctx = make_ctx()

    await section.build(ctx)

    assert [c["chunk_id"] for c in ctx.state["retrieved_chunks"]] == ["good"]


@pytest.mark.asyncio
async def test_build_does_not_touch_state_when_no_hits(embedder):
    section, _ = make_section(embedder, hits=[])
    ctx = make_ctx(query="x")

    await section.build(ctx)

    assert ctx.state == {"query": "x"}


def test_default_priority_sits_between_fewshot_and_persona(embedder):
    section, _ = make_section(embedder)

    assert 40 < section.priority < 100


# ---------- from_config ----------


def make_deps(embedder="default", chunk_store="default"):
    return SimpleNamespace(
        embedder=FakeEmbedder() if embedder == "default" else embedder,
        stores=SimpleNamespace(
            chunk_store=FakeChunkStore() if chunk_store == "default" else chunk_store
        ),
    )


def test_from_config_wires_deps_and_params():
    deps = make_deps()

    section = RAGSection.from_config({"k": 6, "min_score": 0.3}, deps)

    assert section.embedder is deps.embedder
    assert section.chunk_store is deps.stores.chunk_store
    assert section.k == 6
    assert section.min_score == 0.3


def test_from_config_uses_defaults_when_params_empty():
    section = RAGSection.from_config({}, make_deps())

    assert section.k == 4
    assert section.min_score == 0.0


def test_from_config_unknown_param_raises():
    with pytest.raises(ValueError, match="Unknown RAGSection param"):
        RAGSection.from_config({"top_k": 3}, make_deps())


def test_from_config_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        RAGSection.from_config({}, make_deps(embedder=None))
