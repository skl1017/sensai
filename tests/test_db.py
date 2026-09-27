"""
Tests for db.py — the SQLModel + sqlite-vec engine/schema setup.

Since db.py runs its setup as *module-level* code (engine creation, PRAGMA/extension
loading, SQLModel.metadata.create_all, virtual table creation) as soon as it's
imported, these tests re-import a fresh copy of the module for each test in an
isolated temp directory. That gives every test a clean "data.db" and avoids
polluting (or depending on) the real project database, without needing to touch
db.py itself.
"""

import importlib
import json
import struct
import sys
from pathlib import Path

import pytest
import sqlite_vec
from sqlalchemy import inspect, text
from sqlmodel import SQLModel

# db.py lives in config/ and imports its sibling models.py with a bare
# `from models import ...` (not `from config.models import ...` or
# `from .models import ...`). That only works if config/ itself is on
# sys.path, so we add it here rather than importing "config.db".
# Assumes this test file lives in <project_root>/tests/ and the code lives
# in <project_root>/config/ — adjust CONFIG_DIR below if your layout differs.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "db"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))


@pytest.fixture()
def db_module(tmp_path, monkeypatch):
    """Import a fresh `db` module with data.db created inside tmp_path."""
    monkeypatch.chdir(tmp_path)

    # Make sure we get a brand-new import (module-level setup code re-runs).
    for mod_name in ("db", "models"):
        sys.modules.pop(mod_name, None)

    # SQLModel.metadata (and its declarative registry) is a shared, global
    # object that survives even after "models" is removed from sys.modules.
    # Clear it so re-importing models.py doesn't collide with Table objects
    # left over from a previous test's import.
    SQLModel.metadata.clear()

    module = importlib.import_module("db")

    yield module

    module.engine.dispose()
    sys.modules.pop("db", None)
    sys.modules.pop("models", None)


def test_data_db_file_is_created(db_module, tmp_path):
    assert (tmp_path / "data.db").exists()


def test_foreign_keys_pragma_is_enabled(db_module):
    with db_module.engine.connect() as conn:
        result = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
    assert result == 1


def test_sqlite_vec_extension_is_loaded(db_module):
    """If sqlite_vec.load() succeeded, vec_version() must be callable."""
    with db_module.engine.connect() as conn:
        version = conn.exec_driver_sql("SELECT vec_version()").scalar()
    assert version


def test_load_extension_is_disabled_after_setup(db_module):
    """enable_load_extension(False) at the end of the hook should stick,
    so further ad-hoc extension loading from raw SQL is blocked."""
    with db_module.engine.connect() as conn:
        raw_conn = conn.connection.dbapi_connection
        with pytest.raises(Exception):
            raw_conn.execute("SELECT load_extension('does_not_matter')")


def test_sqlmodel_tables_are_created(db_module):
    inspector = inspect(db_module.engine)
    existing_tables = set(inspector.get_table_names())

    expected_tables = set(SQLModel.metadata.tables.keys())
    assert expected_tables, "SQLModel.metadata has no tables — did models import?"
    assert expected_tables.issubset(existing_tables)


def test_chunk_vectors_virtual_table_exists(db_module):
    with db_module.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'chunk_vectors'"
            )
        ).fetchone()

    assert row is not None, "chunk_vectors virtual table was not created"
    name, sql = row
    assert name == "chunk_vectors"
    assert "vec0" in sql


def test_create_all_and_virtual_table_creation_are_idempotent(db_module):
    """Re-running module setup (CREATE TABLE IF NOT EXISTS / vec0) must not error."""
    SQLModel.metadata.create_all(db_module.engine)
    with db_module.engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[768]
            )
            """
        )
    # No exception -> success. Sanity-check the table is still there.
    with db_module.engine.connect() as conn:
        count = conn.exec_driver_sql(
            "SELECT count(*) FROM sqlite_master WHERE name = 'chunk_vectors'"
        ).scalar()
    assert count == 1


def _to_blob(vector):
    """Serialize a list of floats the way sqlite-vec expects (matches
    sqlite_vec.serialize_float32, reimplemented here to avoid depending on
    a specific helper name across sqlite-vec versions)."""
    return struct.pack(f"{len(vector)}f", *vector)


def test_insert_and_query_chunk_vector(db_module):
    embedding = [0.1] * 768

    with db_module.engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            ("chunk-1", _to_blob(embedding)),
        )

    with db_module.engine.connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT chunk_id FROM chunk_vectors WHERE chunk_id = ?",
            ("chunk-1",),
        ).fetchone()

    assert row is not None
    assert row[0] == "chunk-1"


def test_nearest_neighbor_query_on_chunk_vectors(db_module):
    with db_module.engine.begin() as conn:
        for i in range(3):
            vector = [float(i)] * 768
            conn.exec_driver_sql(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (f"chunk-{i}", _to_blob(vector)),
            )

    query_vector = _to_blob([0.0] * 768)
    with db_module.engine.connect() as conn:
        rows = conn.exec_driver_sql(
            """
            SELECT chunk_id, distance
            FROM chunk_vectors
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT 1
            """,
            (query_vector,),
        ).fetchall()

    assert rows, "expected at least one nearest-neighbor match"
    assert rows[0][0] == "chunk-0"
