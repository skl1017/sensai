"""Tests pour storage/ingest.py

Dépendances : pip install pytest pytest-asyncio
Lancer      : pytest -v tests/test_ingest.py

Si tu n'as pas de config pytest-asyncio, ajoute dans pyproject.toml :
    [tool.pytest.ini_options]
    asyncio_mode = "auto"
(les marqueurs @pytest.mark.asyncio ci-dessous fonctionnent aussi en mode strict).
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from config.db.models import IngestedFile
from storage import ingest as ingest_mod
from storage.ingest import FileHash, diff, embed_file, file_hash, read_blocks


# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Embedder factice : un vecteur [len(texte)] par texte, et on garde les appels."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


class FakeChunkStore:
    """ChunkStore factice : enregistre les suppressions et les ajouts."""

    def __init__(self) -> None:
        self.deleted: list[int] = []
        self.added: list[tuple[list, list]] = []

    async def delete_by_source(self, source_id: int) -> None:
        self.deleted.append(source_id)

    async def add_chunks(self, chunks: list, embeddings: list) -> None:
        self.added.append((chunks, embeddings))

    def added_source_ids(self) -> set:
        return {c.source_id for chunks, _ in self.added for c in chunks}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Base SQLite temporaire, injectée à la place de l'engine global."""
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(ingest_mod, "engine", eng)
    return eng


@pytest.fixture
def docs(tmp_path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    return d


@pytest.fixture
def store(monkeypatch) -> FakeChunkStore:
    """Remplace ChunkStore, SqliteVectorStorage et OllamaEmbedder par des fakes."""
    fake = FakeChunkStore()
    monkeypatch.setattr(ingest_mod, "SqliteVectorStorage", lambda engine: object())
    monkeypatch.setattr(ingest_mod, "ChunkStore", lambda engine, vector_store: fake)
    monkeypatch.setattr(ingest_mod, "OllamaEmbedder", FakeEmbedder)
    return fake


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def register(engine, path: Path, content_hash: str | None = None) -> None:
    """Insère directement un fichier dans la table ingested_files."""
    with Session(engine) as s:
        s.add(
            IngestedFile(
                file_path=path.as_posix(),
                content_hash=content_hash or file_hash(path),
                file_name=path.name,
                last_modified=datetime.now(UTC),
            )
        )
        s.commit()


def all_rows(engine) -> list[IngestedFile]:
    with Session(engine) as s:
        return list(s.exec(select(IngestedFile)).all())


# --------------------------------------------------------------------------- #
# read_blocks
# --------------------------------------------------------------------------- #
def test_read_blocks_small_file_gives_single_block(tmp_path):
    f = write(tmp_path / "a.txt", "hello")
    assert list(read_blocks(str(f), block_size=500, overlap=50)) == [("hello", 0)]


def test_read_blocks_empty_file_gives_nothing(tmp_path):
    f = write(tmp_path / "empty.txt", "")
    assert list(read_blocks(str(f))) == []


def test_read_blocks_overlap_exact_content(tmp_path):
    f = write(tmp_path / "alpha.txt", "abcdefghijklmnopqrstuvwxyz")
    blocks = list(read_blocks(str(f), block_size=10, overlap=3))
    assert blocks == [
        ("abcdefghij", 0),
        ("hijklmnopq", 1),
        ("opqrstuvwx", 2),
        ("vwxyz", 3),
    ]


def test_read_blocks_consecutive_blocks_share_overlap(tmp_path):
    text = "".join(chr(97 + i % 26) for i in range(1000))
    f = write(tmp_path / "long.txt", text)
    overlap = 20
    blocks = [b for b, _ in read_blocks(str(f), block_size=100, overlap=overlap)]
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        assert prev[-overlap:] == nxt[:overlap]


def test_read_blocks_can_rebuild_original_text(tmp_path):
    text = "".join(chr(97 + i % 26) for i in range(1234))
    f = write(tmp_path / "long.txt", text)
    overlap = 15
    blocks = [b for b, _ in read_blocks(str(f), block_size=100, overlap=overlap)]
    rebuilt = blocks[0] + "".join(b[overlap:] for b in blocks[1:])
    assert rebuilt == text


def test_read_blocks_indexes_are_sequential(tmp_path):
    f = write(tmp_path / "x.txt", "x" * 2000)
    indexes = [i for _, i in read_blocks(str(f), block_size=100, overlap=10)]
    assert indexes == list(range(len(indexes)))


def test_read_blocks_blocks_never_exceed_block_size(tmp_path):
    f = write(tmp_path / "x.txt", "x" * 2000)
    assert all(len(b) <= 100 for b, _ in read_blocks(str(f), block_size=100, overlap=10))


@pytest.mark.parametrize("overlap", [500, 501])
def test_read_blocks_invalid_overlap_raises(tmp_path, overlap):
    f = write(tmp_path / "a.txt", "hello")
    with pytest.raises(ValueError):
        list(read_blocks(str(f), block_size=500, overlap=overlap))


# --------------------------------------------------------------------------- #
# file_hash
# --------------------------------------------------------------------------- #
def test_file_hash_matches_sha256(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"contenu" * 10_000)
    assert file_hash(f) == hashlib.sha256(b"contenu" * 10_000).hexdigest()


def test_file_hash_independent_of_chunk_size(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc" * 1000)
    assert file_hash(f, chunk_size=7) == file_hash(f, chunk_size=65536)


def test_file_hash_changes_with_content(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v1")
    h1 = file_hash(f)
    f.write_text("v2")
    assert file_hash(f) != h1


# --------------------------------------------------------------------------- #
# embed_file
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embed_file_returns_one_embedding_per_chunk(tmp_path):
    f = write(tmp_path / "a.txt", "lorem ipsum " * 200)
    embedder = FakeEmbedder()

    chunks, embeddings = await embed_file(1, str(f), embedder)

    assert len(chunks) > 1
    assert len(chunks) == len(embeddings)
    # l'embedding i correspond bien au texte du chunk i
    assert [e[0] for e in embeddings] == [float(len(c.text)) for c in chunks]


@pytest.mark.asyncio
async def test_embed_file_sets_chunk_metadata(tmp_path):
    f = write(tmp_path / "a.txt", "lorem ipsum " * 200)

    chunks, _ = await embed_file(42, str(f), FakeEmbedder())

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source_id == 42 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # ids uniques
    assert all(c.doc_category is None for c in chunks)
    assert all(datetime.fromisoformat(c.created_at) for c in chunks)


@pytest.mark.asyncio
async def test_embed_file_calls_embedder_once_with_all_texts(tmp_path):
    f = write(tmp_path / "a.txt", "lorem ipsum " * 200)
    embedder = FakeEmbedder()

    chunks, _ = await embed_file(1, str(f), embedder)

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [c.text for c in chunks]


@pytest.mark.asyncio
async def test_embed_file_empty_file(tmp_path):
    f = write(tmp_path / "empty.txt", "")

    chunks, embeddings = await embed_file(1, str(f), FakeEmbedder())

    assert chunks == []
    assert embeddings == []


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def test_diff_new_file_is_added(engine, docs):
    new = write(docs / "new.txt", "nouveau")

    to_remove, to_add = diff(str(docs))

    assert to_remove == []
    assert to_add == [FileHash(new.as_posix(), file_hash(new))]


def test_diff_unchanged_file_is_ignored(engine, docs):
    f = write(docs / "same.txt", "inchangé")
    register(engine, f)

    to_remove, to_add = diff(str(docs))

    assert to_remove == []
    assert to_add == []


def test_diff_deleted_file_is_removed(engine, docs):
    ghost = docs / "gone.txt"  # n'existe pas sur le disque
    register(engine, ghost, content_hash="deadbeef")

    to_remove, to_add = diff(str(docs))

    assert to_remove == [ghost.as_posix()]
    assert to_add == []


def test_diff_modified_file_is_removed_then_added_once(engine, docs):
    f = write(docs / "mod.txt", "version 1")
    register(engine, f)
    write(f, "version 2")
    new_hash = file_hash(f)

    to_remove, to_add = diff(str(docs))

    assert to_remove == [f.as_posix()]
    # doit être réajouté UNE fois, avec le NOUVEAU hash uniquement
    assert to_add == [FileHash(f.as_posix(), new_hash)]


def test_diff_mixed_scenario(engine, docs):
    same = write(docs / "same.txt", "same")
    mod = write(docs / "mod.txt", "old")
    gone = docs / "gone.txt"
    register(engine, same)
    register(engine, mod)
    register(engine, gone, content_hash="deadbeef")
    write(mod, "new")
    new = write(docs / "sub" / "new.txt", "brand new") if (docs / "sub").mkdir() is None else None

    to_remove, to_add = diff(str(docs))

    assert set(to_remove) == {gone.as_posix(), mod.as_posix()}
    assert {p.path for p in to_add} == {mod.as_posix(), new.as_posix()}


def test_diff_empty_directory_and_empty_db(engine, docs):
    assert diff(str(docs)) == ([], [])


# --------------------------------------------------------------------------- #
# ingest (bout en bout, avec fakes)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ingest_first_run_indexes_all_files(engine, docs, store):
    write(docs / "a.txt", "alpha " * 300)
    write(docs / "b.txt", "beta " * 300)

    await ingest_mod.ingest(str(docs))

    rows = all_rows(engine)
    assert {r.file_name for r in rows} == {"a.txt", "b.txt"}
    assert store.deleted == []
    assert store.added_source_ids() == {r.id for r in rows}


@pytest.mark.asyncio
async def test_ingest_second_run_without_changes_does_nothing(engine, docs, store):
    write(docs / "a.txt", "alpha " * 300)
    await ingest_mod.ingest(str(docs))
    store.added.clear()

    await ingest_mod.ingest(str(docs))

    assert len(all_rows(engine)) == 1
    assert store.deleted == []
    assert store.added == []


@pytest.mark.asyncio
async def test_ingest_modified_file_is_reindexed(engine, docs, store):
    a = write(docs / "a.txt", "alpha " * 300)
    write(docs / "b.txt", "beta " * 300)
    await ingest_mod.ingest(str(docs))
    old_a = next(r for r in all_rows(engine) if r.file_name == "a.txt")
    b_id = next(r.id for r in all_rows(engine) if r.file_name == "b.txt")
    store.added.clear()

    write(a, "alpha modifié " * 300)
    await ingest_mod.ingest(str(docs))

    rows = all_rows(engine)
    new_a = [r for r in rows if r.file_name == "a.txt"]
    assert len(rows) == 2  # pas de doublon
    assert len(new_a) == 1
    assert new_a[0].content_hash == file_hash(a)
    assert store.deleted == [old_a.id]  # anciens chunks supprimés
    assert store.added_source_ids() == {new_a[0].id}  # seul a.txt est réindexé
    assert b_id not in store.added_source_ids()


@pytest.mark.asyncio
async def test_ingest_deleted_file_is_cleaned_up(engine, docs, store):
    a = write(docs / "a.txt", "alpha " * 300)
    write(docs / "b.txt", "beta " * 300)
    await ingest_mod.ingest(str(docs))
    a_id = next(r.id for r in all_rows(engine) if r.file_name == "a.txt")
    store.added.clear()

    a.unlink()
    await ingest_mod.ingest(str(docs))

    assert [r.file_name for r in all_rows(engine)] == ["b.txt"]
    assert store.deleted == [a_id]
    assert store.added == []


@pytest.mark.asyncio
async def test_ingest_empty_directory(engine, docs, store):
    await ingest_mod.ingest(str(docs))

    assert all_rows(engine) == []
    assert store.deleted == []
    assert store.added == []
