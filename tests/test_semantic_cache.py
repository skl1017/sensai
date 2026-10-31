import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from storage.semantic_cache import SemanticCache
from storage.vector_storage import SqliteVectorStorage

DIM = 4
SCOPE = "persona:rh"


class FakeClock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    # StaticPool : une seule connexion partagée, car les accès SQL passent par asyncio.to_thread
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    vectors = SqliteVectorStorage(
        engine, embedding_dim=DIM, table="cache_vectors", distance_metric="cosine"
    )  # avant tout autre usage de l'engine
    SQLModel.metadata.create_all(engine)
    return SemanticCache(engine, vectors, clock=clock)


@pytest.mark.asyncio
async def test_empty_cache_returns_none(cache):
    assert await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None) is None


@pytest.mark.asyncio
async def test_same_vector_returns_response_with_similarity_one(cache):
    await cache.store(SCOPE, "q", [1.0, 0.0, 0.0, 0.0], "réponse")

    hit = await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None)

    assert hit is not None
    response, similarity = hit.response, hit.similarity
    assert hit.question == "q"
    assert response == "réponse"
    assert similarity == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_lookup_returns_closest_of_two_entries(cache):
    await cache.store(SCOPE, "far", [0.0, 0.0, 0.0, 1.0], "loin")
    await cache.store(SCOPE, "close", [0.9, 0.1, 0.0, 0.0], "proche")

    hit = await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None)

    assert hit is not None
    assert hit.response == "proche"
    assert 0.9 < hit.similarity < 1.0


@pytest.mark.asyncio
async def test_other_scope_is_never_returned(cache):
    await cache.store("persona:tech", "q", [1.0, 0.0, 0.0, 0.0], "autre scope")

    assert await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None) is None


@pytest.mark.asyncio
async def test_closer_entry_from_other_scope_does_not_hide_own_scope(cache):
    # l'entrée d'un autre scope est l'exacte voisine : elle ne doit pas masquer la nôtre
    await cache.store("persona:tech", "q", [1.0, 0.0, 0.0, 0.0], "autre scope")
    await cache.store(SCOPE, "q", [0.9, 0.1, 0.0, 0.0], "mon scope")

    hit = await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None)

    assert hit is not None
    assert hit.response == "mon scope"


@pytest.mark.asyncio
async def test_expired_entry_is_not_returned(cache, clock):
    await cache.store(SCOPE, "q", [1.0, 0.0, 0.0, 0.0], "vieux")
    clock.now += 61

    assert await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=60) is None


@pytest.mark.asyncio
async def test_entry_within_ttl_is_returned(cache, clock):
    await cache.store(SCOPE, "q", [1.0, 0.0, 0.0, 0.0], "récent")
    clock.now += 30

    hit = await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=60)

    assert hit is not None
    assert hit.response == "récent"


@pytest.mark.asyncio
async def test_ttl_none_never_expires(cache, clock):
    await cache.store(SCOPE, "q", [1.0, 0.0, 0.0, 0.0], "éternel")
    clock.now += 10_000_000

    hit = await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None)

    assert hit is not None
    assert hit.response == "éternel"


@pytest.mark.asyncio
async def test_purge_expired_removes_only_expired_entries(cache, clock):
    await cache.store(SCOPE, "old", [1.0, 0.0, 0.0, 0.0], "vieux")
    clock.now += 100
    await cache.store(SCOPE, "new", [0.0, 1.0, 0.0, 0.0], "récent")

    removed = await cache.purge_expired(ttl_seconds=50)

    assert removed == 1
    assert await cache.lookup(SCOPE, [1.0, 0.0, 0.0, 0.0], ttl_seconds=None) is not None
    hit = await cache.lookup(SCOPE, [0.0, 1.0, 0.0, 0.0], ttl_seconds=None)
    assert hit is not None
    assert hit.response == "récent"


@pytest.mark.asyncio
async def test_purge_expired_deletes_vectors_too(cache, clock):
    await cache.store(SCOPE, "old", [1.0, 0.0, 0.0, 0.0], "vieux")
    clock.now += 100

    await cache.purge_expired(ttl_seconds=50)

    with cache._engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM cache_vectors").scalar() == 0


@pytest.mark.asyncio
async def test_purge_expired_on_empty_cache_returns_zero(cache):
    assert await cache.purge_expired(ttl_seconds=10) == 0
