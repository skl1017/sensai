from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from config.db.models import Chunk, IngestedFile
from core.storage import IVectorStorage
from storage.chunk_store import ChunkStore
from storage.ingest import ChunkData
from storage.vector_storage import SqliteVectorStorage

DIM = 4


class SearchableFakeVectors(IVectorStorage):
    """Vector storage en mémoire dont les résultats de search sont préparés à l'avance."""

    def __init__(self, results=None):
        self.vectors: dict[str, list[float]] = {}
        self.results = results or []
        self.search_calls: list[tuple] = []

    async def add(self, chunk_id, embedding):
        self.vectors[chunk_id] = embedding

    async def search(self, collection, vector, k, filters=None):
        self.search_calls.append((collection, vector, k))
        return self.results

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


def hit(chunk_id: str, distance: float = 0.5):
    # Même forme que VectorSearchResult(chunk_id, distance)
    return SimpleNamespace(chunk_id=chunk_id, distance=distance)


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


# ---------- search (fake) ----------


@pytest.mark.asyncio
async def test_search_returns_chunks_in_vector_order(engine):
    vectors = SearchableFakeVectors(results=[hit("c2", 0.1), hit("c1", 0.6)])
    store = ChunkStore(engine, vectors)
    await store.add_chunks(
        [make_chunk("c1"), make_chunk("c2", index=1)], [[1.0] * DIM, [2.0] * DIM]
    )

    results = await store.search([1.0] * DIM, k=2)

    assert [c.id for c, _ in results] == ["c2", "c1"]


@pytest.mark.asyncio
async def test_search_converts_distance_to_similarity(engine):
    vectors = SearchableFakeVectors(results=[hit("c2", 0.1), hit("c1", 0.6)])
    store = ChunkStore(engine, vectors)
    await store.add_chunks(
        [make_chunk("c1"), make_chunk("c2", index=1)], [[1.0] * DIM, [2.0] * DIM]
    )

    results = await store.search([1.0] * DIM, k=2)

    # L2 sur vecteurs unitaires : similarité = 1 - d² / 2, le plus proche a le meilleur score
    assert [score for _, score in results] == pytest.approx([0.995, 0.82])


@pytest.mark.asyncio
async def test_search_returns_full_chunk_fields(engine):
    vectors = SearchableFakeVectors(results=[hit("c1")])
    store = ChunkStore(engine, vectors)
    await store.add_chunks([make_chunk("c1", index=3, category="Technique")], [[1.0] * DIM])

    ((chunk, _),) = await store.search([1.0] * DIM, k=1)

    assert chunk.text == "texte c1"
    assert chunk.source_id == "src1"
    assert chunk.chunk_index == 3
    assert chunk.doc_category == "Technique"


@pytest.mark.asyncio
async def test_search_returns_source_file_name(engine):
    with Session(engine) as session:
        source = IngestedFile(
            file_name="rh.pdf",
            file_path="/docs/rh.pdf",
            last_modified=datetime(2026, 3, 14, 9, 30),
            content_hash="h",
        )
        session.add(source)
        session.commit()
        session.refresh(source)
        source_id = source.id
    store = ChunkStore(engine, SearchableFakeVectors(results=[hit("c1")]))
    await store.add_chunks([make_chunk("c1", source_id=source_id)], [[1.0] * DIM])

    ((chunk, _),) = await store.search([1.0] * DIM, k=1)

    assert chunk.source_name == "rh.pdf"
    assert chunk.document_date == "2026-03-14"


@pytest.mark.asyncio
async def test_search_source_name_is_none_without_ingested_file(engine):
    store = ChunkStore(engine, SearchableFakeVectors(results=[hit("c1")]))
    await store.add_chunks(
        [make_chunk("c1")], [[1.0] * DIM]
    )  # source_id "src1", pas de ligne IngestedFile

    ((chunk, _),) = await store.search([1.0] * DIM, k=1)

    assert chunk.source_name is None
    assert chunk.document_date is None


@pytest.mark.asyncio
async def test_search_without_hits_returns_empty_list(engine):
    store = ChunkStore(engine, SearchableFakeVectors(results=[]))

    assert await store.search([1.0] * DIM, k=3) == []


@pytest.mark.asyncio
async def test_search_ignores_vector_hits_missing_from_database(engine):
    vectors = SearchableFakeVectors(results=[hit("ghost", 0.9), hit("c1", 0.5)])
    store = ChunkStore(engine, vectors)
    await store.add_chunks([make_chunk("c1")], [[1.0] * DIM])

    results = await store.search([1.0] * DIM, k=2)

    assert [c.id for c, _ in results] == ["c1"]


@pytest.mark.asyncio
async def test_search_queries_chunk_vectors_collection_with_k(engine):
    vectors = SearchableFakeVectors()
    store = ChunkStore(engine, vectors)

    await store.search([0.1, 0.2, 0.3, 0.4], k=7)

    assert vectors.search_calls == [("chunk_vectors", [0.1, 0.2, 0.3, 0.4], 7)]


def test_file_names_lists_files_with_chunks_once(engine):
    store = ChunkStore(engine, SearchableFakeVectors())
    with Session(engine) as session:
        for name in ("rh.pdf", "cv.pdf", "vide.pdf"):
            session.add(
                IngestedFile(
                    file_name=name,
                    file_path=f"/docs/{name}",
                    last_modified=datetime(2026, 3, 14),
                    content_hash=name,
                )
            )
        session.commit()
        ids = {f.file_name: f.id for f in session.exec(select(IngestedFile)).all()}
        for chunk_id, name in (("c1", "rh.pdf"), ("c2", "rh.pdf"), ("c3", "cv.pdf")):
            session.add(
                Chunk(
                    id=chunk_id,
                    text="t",
                    source_id=ids[name],
                    chunk_index=0,
                    created_at="2026-01-01T00:00:00+00:00",
                )
            )
        session.commit()

    assert store.file_names() == ["cv.pdf", "rh.pdf"]


# ---------- intégration avec le vrai sqlite-vec ----------


@pytest.fixture
def real_store():
    engine = create_engine("sqlite://")
    vectors = SqliteVectorStorage(engine, embedding_dim=DIM)  # avant tout autre usage de l'engine
    SQLModel.metadata.create_all(engine)
    return ChunkStore(engine, vectors)


@pytest.mark.asyncio
async def test_integration_search_returns_closest_chunk_first(real_store):
    await real_store.add_chunks(
        [make_chunk("close"), make_chunk("far", index=1)],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )

    results = await real_store.search([1.0, 0.0, 0.0, 0.0], k=5)

    assert [c.id for c, _ in results] == ["close", "far"]
    (_, close_score), (_, far_score) = results
    assert close_score > far_score


@pytest.mark.asyncio
async def test_integration_search_respects_k(real_store):
    await real_store.add_chunks(
        [make_chunk("a"), make_chunk("b", index=1), make_chunk("c", index=2)],
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )

    results = await real_store.search([1.0, 0.0, 0.0, 0.0], k=2)

    assert [c.id for c, _ in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_integration_search_after_delete_by_source_is_empty(real_store):
    await real_store.add_chunks([make_chunk("c1")], [[1.0, 0.0, 0.0, 0.0]])

    await real_store.delete_by_source("src1")

    assert await real_store.search([1.0, 0.0, 0.0, 0.0], k=5) == []


# ---------- filtres optionnels (vrai sqlite-vec) ----------


QUERY = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
async def filtered_store(real_store):
    """rh.pdf (janvier), accord.pdf (mars), notes.txt (mai) et un chunk sans fichier."""
    sources = [
        ("rh.pdf", datetime(2026, 1, 15, 23, 59)),
        ("accord.pdf", datetime(2026, 3, 15)),
        ("notes.txt", datetime(2026, 5, 15)),
    ]
    chunks, embeddings = [], []
    with Session(real_store._engine) as session:
        for position, (name, modified) in enumerate(sources):
            source = IngestedFile(
                file_name=name, file_path=f"/docs/{name}", last_modified=modified, content_hash="h"
            )
            session.add(source)
            session.commit()
            session.refresh(source)
            chunks.append(make_chunk(name, source_id=source.id))
            embeddings.append([1.0, position / 10, 0.0, 0.0])
    chunks.append(make_chunk("orphan"))
    embeddings.append([1.0, 0.0, 0.0, 0.0])
    await real_store.add_chunks(chunks, embeddings)
    return real_store


async def found(store, **filters) -> list[str]:
    return [chunk.id for chunk, _ in await store.search(QUERY, k=10, **filters)]


@pytest.mark.asyncio
async def test_search_without_filter_keeps_every_chunk(filtered_store):
    assert set(await found(filtered_store)) == {"rh.pdf", "accord.pdf", "notes.txt", "orphan"}


@pytest.mark.asyncio
async def test_search_file_name_filter(filtered_store):
    assert await found(filtered_store, file_name="accord.pdf") == ["accord.pdf"]


@pytest.mark.asyncio
async def test_search_filter_returns_k_results_among_allowed_chunks(filtered_store):
    # Les chunks plus proches sont exclus avant la recherche, ils ne prennent pas de place dans k.
    results = await filtered_store.search(QUERY, k=1, file_name="notes.txt")

    assert [chunk.id for chunk, _ in results] == ["notes.txt"]


@pytest.mark.asyncio
async def test_search_date_from_is_inclusive(filtered_store):
    assert await found(filtered_store, date_from=date(2026, 3, 15)) == ["accord.pdf", "notes.txt"]


@pytest.mark.asyncio
async def test_search_date_to_includes_the_whole_day(filtered_store):
    assert await found(filtered_store, date_to=date(2026, 1, 15)) == ["rh.pdf"]


@pytest.mark.asyncio
async def test_search_date_range(filtered_store):
    period = {"date_from": date(2026, 2, 1), "date_to": date(2026, 4, 1)}

    assert await found(filtered_store, **period) == ["accord.pdf"]


@pytest.mark.asyncio
async def test_search_combines_file_name_and_dates(filtered_store):
    assert await found(filtered_store, file_name="rh.pdf", date_from=date(2026, 2, 1)) == []


@pytest.mark.asyncio
async def test_search_unknown_file_name_returns_nothing(filtered_store):
    assert await found(filtered_store, file_name="absent.pdf") == []
