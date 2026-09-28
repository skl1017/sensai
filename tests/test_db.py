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
import sqlite3
import struct
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlmodel import SQLModel

import importlib
import sys

import pytest
from sqlmodel import SQLModel

MODULES = ("config.db.db", "config.db.models")


def _purge():
    for name in MODULES:
        sys.modules.pop(name, None)
    SQLModel.metadata.clear()


@pytest.fixture()
def db_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _purge()
    module = importlib.import_module("config.db.db")
    yield module
    module.engine.dispose()
    _purge()

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
        with pytest.raises(sqlite3.OperationalError):
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
