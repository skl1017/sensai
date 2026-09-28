import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from config.db.models import Chunk
from core.storage import IVectorStorage
from storage.chunk_store import ChunkStore
from storage.ingest import ChunkData
from storage.vector_storage import SqliteVectorStorage

DIM = 4


class FakeVectorStorage(IVectorStorage):
    """Vector storage en mémoire : permet de tester ChunkStore isolément."""

    def __init__(self):
        self.vectors: dict[str, list[float]] = {}

    async def add(self, chunk_id, embedding):
        self.vectors[chunk_id] = embedding

    async def search(self, collection, vector, k, filters=None):
        raise NotImplementedError

    async def delete(self, chunk_id):
        self.vectors.pop(chunk_id, None)


def make_chunk(chunk_id: str, source_id: str = "src1", index: int = 0, category: str | None = "RH"):
    return ChunkData(
        id=chunk_id,
        text=f"texte {chunk_id}",
        source_id=source_id,
        chunk_index=index,
        doc_category=category,
        created_at="2026-01-01T00:00:00+00:00",
    )


def rows(engine) -> list[Chunk]:
    with Session(engine) as session:
        return session.exec(select(Chunk)).all()


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def fake_vectors():
    return FakeVectorStorage()


@pytest.fixture
def store(engine, fake_vectors):
    return ChunkStore(engine, fake_vectors)


# ---------- add_chunks ----------


@pytest.mark.asyncio
async def test_add_chunks_persists_rows_with_all_fields(store, engine):
    await store.add_chunks([make_chunk("c1", index=3, category="Technique")], [[1.0] * DIM])

    (row,) = rows(engine)
    assert row.id == "c1"
    assert row.text == "texte c1"
    assert row.source_id == "src1"
    assert row.chunk_index == 3
    assert row.doc_category == "Technique"
    assert row.created_at == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_add_chunks_writes_matching_vectors(store, fake_vectors):
    chunks = [make_chunk("c1"), make_chunk("c2", index=1)]
    embeddings = [[1.0] * DIM, [2.0] * DIM]

    await store.add_chunks(chunks, embeddings)

    assert fake_vectors.vectors == {"c1": [1.0] * DIM, "c2": [2.0] * DIM}


@pytest.mark.asyncio
async def test_add_chunks_with_empty_lists_does_nothing(store, engine, fake_vectors):
    await store.add_chunks([], [])

    assert rows(engine) == []
    assert fake_vectors.vectors == {}


@pytest.mark.asyncio
async def test_add_chunks_length_mismatch_raises_and_persists_nothing(store, engine, fake_vectors):
    chunks = [make_chunk("c1"), make_chunk("c2", index=1)]

    with pytest.raises(ValueError):
        await store.add_chunks(chunks, [[1.0] * DIM])  # 2 chunks, 1 embedding

    # Ne doit pas laisser de chunks orphelins (sans vecteur) en base
    assert rows(engine) == []
    assert fake_vectors.vectors == {}


# ---------- delete_by_source ----------


@pytest.mark.asyncio
async def test_delete_by_source_removes_rows_and_vectors(store, engine, fake_vectors):
    await store.add_chunks(
        [make_chunk("c1"), make_chunk("c2", index=1)], [[1.0] * DIM, [2.0] * DIM]
    )

    await store.delete_by_source("src1")

    assert rows(engine) == []
    assert fake_vectors.vectors == {}


@pytest.mark.asyncio
async def test_delete_by_source_only_affects_targeted_source(store, engine, fake_vectors):
    await store.add_chunks(
        [make_chunk("a1", source_id="A"), make_chunk("b1", source_id="B")],
        [[1.0] * DIM, [2.0] * DIM],
    )

    await store.delete_by_source("A")

    assert [r.id for r in rows(engine)] == ["b1"]
    assert list(fake_vectors.vectors) == ["b1"]


@pytest.mark.asyncio
async def test_delete_by_source_unknown_source_does_not_raise(store, engine):
    await store.delete_by_source("inconnu")

    assert rows(engine) == []


# ---------- intégration avec le vrai sqlite-vec ----------


@pytest.fixture
def real_store():
    engine = create_engine("sqlite://")
    vectors = SqliteVectorStorage(engine, embedding_dim=DIM)  # avant tout autre usage de l'engine
    SQLModel.metadata.create_all(engine)
    return ChunkStore(engine, vectors), vectors


@pytest.mark.asyncio
async def test_integration_added_chunks_are_searchable(real_store):
    store, vectors = real_store
    await store.add_chunks(
        [make_chunk("close"), make_chunk("far", index=1)],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )

    results = await vectors.search("chunk_vectors", [1.0, 0.0, 0.0, 0.0], k=5)

    assert [r.chunk_id for r in results] == ["close", "far"]


@pytest.mark.asyncio
async def test_integration_delete_by_source_removes_from_vector_search(real_store):
    store, vectors = real_store
    await store.add_chunks([make_chunk("c1")], [[1.0, 0.0, 0.0, 0.0]])

    await store.delete_by_source("src1")

    results = await vectors.search("chunk_vectors", [1.0, 0.0, 0.0, 0.0], k=5)
    assert results == []


@pytest.mark.asyncio
async def test_integration_reingestion_replaces_chunks(real_store):
    store, vectors = real_store
    await store.add_chunks(
        [make_chunk("c1"), make_chunk("c2", index=1)],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    # même pattern que IngestionService : delete puis add
    await store.delete_by_source("src1")
    await store.add_chunks([make_chunk("c1")], [[0.0, 0.0, 1.0, 0.0]])

    results = await vectors.search("chunk_vectors", [0.0, 0.0, 1.0, 0.0], k=5)
    assert [r.chunk_id for r in results] == ["c1"]
    assert len(rows(store._engine)) == 1
