import re
import struct

from sqlalchemy import event

from core.storage import IVectorStorage, VectorSearchResult

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DISTANCE_METRICS = ("l2", "cosine", "l1")


class SqliteVectorStorage(IVectorStorage):
    def __init__(
        self,
        engine,
        embedding_dim: int = 768,
        table: str = "chunk_vectors",
        distance_metric: str = "l2",
    ):
        if not _TABLE_NAME_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        if distance_metric not in _DISTANCE_METRICS:
            raise ValueError(
                f"invalid distance_metric: {distance_metric!r} "
                f"(expected one of {_DISTANCE_METRICS})"
            )
        self._engine = engine
        self.embedding_dim = embedding_dim
        self.table = table
        self.distance_metric = distance_metric
        self._register_vec_extension()
        self._ensure_table()

    def _register_vec_extension(self) -> None:
        @event.listens_for(self._engine, "connect")
        def load_vec(dbapi_conn, connection_record):
            dbapi_conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

    def _ensure_table(self) -> None:
        embedding_column = f"FLOAT[{self.embedding_dim}]"
        if self.distance_metric != "l2":
            embedding_column += f" distance_metric={self.distance_metric}"
        with self._engine.begin() as conn:
            conn.exec_driver_sql(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding {embedding_column}
                )
            """)

    @staticmethod
    def _serialize(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    async def add(self, chunk_id: str, embedding: list[float]) -> None:
        with self._engine.begin() as conn:
            conn.exec_driver_sql(
                f"DELETE FROM {self.table} WHERE chunk_id = ?",
                (chunk_id,),
            )
            conn.exec_driver_sql(
                f"INSERT INTO {self.table} (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, self._serialize(embedding)),
            )

    async def search(
        self,
        collection: str,
        vector: list[float],
        k: int,
        filters: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        query_blob = self._serialize(vector)

        if filters is not None:
            if not filters:
                return []

            # A one-item IN becomes `=`, which vec0 applies after the k nearest are
            # picked, losing the item unless it's among them: repeating it keeps the IN.
            # A k covering every row instead would fail above vec0's limit of 4096.
            if len(filters) == 1:
                filters = filters * 2
            placeholders = ",".join("?" * len(filters))
            sql = f"""
                SELECT chunk_id, distance
                FROM {collection}
                WHERE chunk_id IN ({placeholders})
                  AND embedding MATCH ?
                  AND k = ?
                ORDER BY distance
            """
            with self._engine.connect() as conn:
                rows = conn.exec_driver_sql(sql, (*filters, query_blob, k)).fetchall()
        else:
            sql = f"""
                SELECT chunk_id, distance
                FROM {collection}
                WHERE embedding MATCH ?
                  AND k = ?
                ORDER BY distance
            """
            with self._engine.connect() as conn:
                rows = conn.exec_driver_sql(sql, (query_blob, k)).fetchall()

        return [VectorSearchResult(chunk_id=row[0], distance=row[1]) for row in rows]

    async def delete(self, chunk_id: str) -> None:
        with self._engine.begin() as conn:
            conn.exec_driver_sql(
                f"DELETE FROM {self.table} WHERE chunk_id = ?",
                (chunk_id,),
            )
