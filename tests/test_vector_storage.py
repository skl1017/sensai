import pytest
from sqlalchemy import create_engine

from storage.vector_storage import SqliteVectorStorage

EMBEDDING_DIM = 4  # petit pour les tests, plus lisible


@pytest.fixture
def storage():
    # base en mémoire : rapide, isolée entre chaque test
    engine = create_engine("sqlite://")
    return SqliteVectorStorage(engine, embedding_dim=EMBEDDING_DIM)


@pytest.mark.asyncio
async def test_add_then_search_returns_the_item(storage):
    await storage.add("chunk_1", [1.0, 0.0, 0.0, 0.0])

    results = await storage.search(
        collection="chunk_vectors",
        vector=[1.0, 0.0, 0.0, 0.0],
        k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_search_orders_by_distance(storage):
    # proche de la query
    await storage.add("close", [1.0, 0.0, 0.0, 0.0])
    # loin de la query
    await storage.add("far", [0.0, 0.0, 0.0, 1.0])

    results = await storage.search(
        collection="chunk_vectors",
        vector=[1.0, 0.0, 0.0, 0.0],
        k=5,
    )

    assert [r.chunk_id for r in results] == ["close", "far"]


@pytest.mark.asyncio
async def test_search_respects_top_k(storage):
    for i in range(5):
        await storage.add(f"chunk_{i}", [float(i), 0.0, 0.0, 0.0])

    results = await storage.search(
        collection="chunk_vectors",
        vector=[0.0, 0.0, 0.0, 0.0],
        k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_with_filters_restricts_to_allowed_ids(storage):
    await storage.add("allowed", [1.0, 0.0, 0.0, 0.0])
    await storage.add("not_allowed", [1.0, 0.0, 0.0, 0.0])  # identique niveau vecteur

    results = await storage.search(
        collection="chunk_vectors",
        vector=[1.0, 0.0, 0.0, 0.0],
        k=5,
        filters=["allowed"],
    )

    assert len(results) == 1
    assert results[0].chunk_id == "allowed"


@pytest.mark.asyncio
async def test_search_with_empty_filters_short_circuits(storage):
    await storage.add("chunk_1", [1.0, 0.0, 0.0, 0.0])

    results = await storage.search(
        collection="chunk_vectors",
        vector=[1.0, 0.0, 0.0, 0.0],
        k=5,
        filters=[],  # liste vide = 0 résultat, sans même toucher la DB
    )

    assert results == []


@pytest.mark.asyncio
async def test_add_replaces_existing_embedding_for_same_id(storage):
    await storage.add("chunk_1", [1.0, 0.0, 0.0, 0.0])
    await storage.add("chunk_1", [0.0, 1.0, 0.0, 0.0])  # même id, nouveau vecteur

    results = await storage.search(
        collection="chunk_vectors",
        vector=[0.0, 1.0, 0.0, 0.0],
        k=5,
    )

    assert len(results) == 1
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)
