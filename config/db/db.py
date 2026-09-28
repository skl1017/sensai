import models  # noqa: F401
from sqlalchemy import create_engine, event
from sqlmodel import SQLModel

from storage.vector_storage import SqliteVectorStorage

engine = create_engine("sqlite:///data.db")


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


vec_storage = SqliteVectorStorage(engine, embedding_dim=768)

SQLModel.metadata.create_all(engine)
