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


@pytest.mark.parametrize("allowed", [["far"], ["far", "farther"]])
async def test_search_with_filters_applies_k_among_allowed_ids_only(storage, allowed):
    await storage.add("close", [1.0, 0.0, 0.0, 0.0])
    await storage.add("far", [0.0, 1.0, 0.0, 0.0])
    await storage.add("farther", [0.0, 0.0, 0.0, 1.0])

    results = await storage.search(
        collection="chunk_vectors", vector=[0.9, 0.4, 0.0, 0.0], k=1, filters=allowed
    )

    assert [r.chunk_id for r in results] == ["far"]


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


# ---------- table et métrique configurables ----------


@pytest.mark.asyncio
async def test_custom_cosine_table_coexists_with_default_chunk_vectors():
    engine = create_engine("sqlite://")
    default = SqliteVectorStorage(engine, embedding_dim=EMBEDDING_DIM)
    cosine = SqliteVectorStorage(
        engine, embedding_dim=EMBEDDING_DIM, table="cache_vectors", distance_metric="cosine"
    )
    await default.add("chunk_1", [1.0, 0.0, 0.0, 0.0])
    await cosine.add("entry_1", [1.0, 0.0, 0.0, 0.0])

    cosine_hits = await cosine.search(collection="cache_vectors", vector=[1.0, 0.0, 0.0, 0.0], k=5)
    default_hits = await default.search(
        collection="chunk_vectors", vector=[1.0, 0.0, 0.0, 0.0], k=5
    )

    assert [h.chunk_id for h in cosine_hits] == ["entry_1"]
    assert [h.chunk_id for h in default_hits] == ["chunk_1"]


@pytest.mark.asyncio
async def test_cosine_distance_is_one_minus_cosine():
    engine = create_engine("sqlite://")
    storage = SqliteVectorStorage(
        engine, embedding_dim=EMBEDDING_DIM, table="cache_vectors", distance_metric="cosine"
    )
    await storage.add("orthogonal", [0.0, 1.0, 0.0, 0.0])

    results = await storage.search(collection="cache_vectors", vector=[1.0, 0.0, 0.0, 0.0], k=1)

    assert results[0].distance == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_filtered_search_is_not_cut_by_closer_ids_outside_filter(storage):
    # sans correction, LIMIT coupe avant le filtre et renvoie []
    await storage.add("outside", [1.0, 0.0, 0.0, 0.0])
    await storage.add("inside", [0.9, 0.1, 0.0, 0.0])

    results = await storage.search(
        collection="chunk_vectors", vector=[1.0, 0.0, 0.0, 0.0], k=1, filters=["inside"]
    )

    assert [r.chunk_id for r in results] == ["inside"]


@pytest.mark.asyncio
async def test_filtered_search_works_above_vec0_k_limit(storage):
    # vec0 refuse k > 4096 : k ne doit pas dépendre du nombre de lignes
    for i in range(4097):
        await storage.add(f"chunk_{i}", [1.0, float(i), 0.0, 0.0])

    results = await storage.search(
        collection="chunk_vectors", vector=[1.0, 0.0, 0.0, 0.0], k=2, filters=["chunk_9", "chunk_4"]
    )

    assert [r.chunk_id for r in results] == ["chunk_4", "chunk_9"]


@pytest.mark.parametrize("table", ["", "1bad", "bad-name", "x; DROP TABLE y", "a b"])
def test_invalid_table_name_raises(table):
    engine = create_engine("sqlite://")

    with pytest.raises(ValueError):
        SqliteVectorStorage(engine, embedding_dim=EMBEDDING_DIM, table=table)


def test_invalid_distance_metric_raises():
    engine = create_engine("sqlite://")

    with pytest.raises(ValueError):
        SqliteVectorStorage(engine, embedding_dim=EMBEDDING_DIM, distance_metric="hamming")
