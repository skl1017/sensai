import models  # noqa: F401
import sqlite_vec
from sqlalchemy import create_engine, event
from sqlmodel import SQLModel

engine = create_engine("sqlite:///data.db")


@event.listens_for(engine, "connect")
def _setup_connection(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA foreign_keys=ON")
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)


SQLModel.metadata.create_all(engine)

with engine.begin() as conn:
    conn.exec_driver_sql("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding FLOAT[768]
        )
    """)
