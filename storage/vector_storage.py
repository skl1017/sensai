import struct

from sqlalchemy import event

from core.storage import IVectorStorage, VectorSearchResult


class SqliteVectorStorage(IVectorStorage):
    def __init__(self, engine, embedding_dim: int = 768):
        self._engine = engine
        self.embedding_dim = embedding_dim
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
        with self._engine.begin() as conn:
            conn.exec_driver_sql(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
            """)

    @staticmethod
    def _serialize(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    async def add(self, chunk_id: str, embedding: list[float]) -> None:
        with self._engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM chunk_vectors WHERE chunk_id = ?",
                (chunk_id,),
            )
            conn.exec_driver_sql(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
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

            placeholders = ",".join("?" * len(filters))
            sql = f"""
                SELECT chunk_id, distance
                FROM {collection}
                WHERE chunk_id IN ({placeholders})
                  AND embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            params = (*filters, query_blob, k)
        else:
            sql = f"""
                SELECT chunk_id, distance
                FROM {collection}
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            params = (query_blob, k)

        with self._engine.connect() as conn:
            rows = conn.exec_driver_sql(sql, params).fetchall()

        return [VectorSearchResult(chunk_id=row[0], distance=row[1]) for row in rows]

    async def delete(self, chunk_id: str) -> None:
        with self._engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM chunk_vectors WHERE chunk_id = ?",
                (chunk_id,),
            )
