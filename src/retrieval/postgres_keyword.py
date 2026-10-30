import json
import os
from typing import Any

from src.retrieval.bm25 import tokenize


class PostgresKeywordIndex:
    """Persistent PostgreSQL full-text keyword index.

    It uses PostgreSQL's built-in English full-text ranking in the deployed app.
    Local evaluation can continue to use the project's custom SQLite BM25 index.
    """

    TABLE = "aiie_search_chunks"

    def __init__(
        self,
        database_url: str | None = None,
        source_type: str = "main",
    ):
        self.database_url = (
            database_url
            or os.getenv("SUPABASE_DB_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        if not self.database_url:
            raise ValueError(
                "Set SUPABASE_DB_URL or DATABASE_URL to use PostgreSQL keyword search."
            )
        if source_type not in {"main", "uploaded"}:
            raise ValueError("source_type must be 'main' or 'uploaded'.")
        self.source_type = source_type
        self._initialize()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL support requires psycopg. Install requirements.txt."
            ) from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _initialize(self) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT to_regclass('public.aiie_search_chunks') AS table_name"
            ).fetchone()
            if exists and exists["table_name"]:
                return

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    source_type TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    payload JSONB NOT NULL,
                    search_vector TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
                    ) STORED,
                    PRIMARY KEY (source_type, chunk_id)
                )
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_aiie_search_vector ON {self.TABLE} USING GIN (search_vector)"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_aiie_search_source_document ON {self.TABLE} (source_type, document_id)"
            )

    @staticmethod
    def _row(source_type: str, document: dict[str, Any]) -> tuple:
        chunk_id = document.get("chunk_id")
        if not chunk_id:
            raise ValueError("Every indexed record must have a chunk_id.")
        document_id = str(document.get("document_id") or document.get("source_id") or "")
        return (
            source_type,
            str(chunk_id),
            document_id,
            str(document.get("title") or ""),
            str(document.get("text") or ""),
            json.dumps(document, ensure_ascii=False),
        )

    def _upsert(self, connection, documents: list[dict[str, Any]], source_type: str) -> None:
        if not documents:
            return
        rows = [self._row(source_type, document) for document in documents]
        with connection.cursor() as cursor:
            cursor.executemany(
                f"""
            INSERT INTO {self.TABLE}
                (source_type, chunk_id, document_id, title, text, payload)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (source_type, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                payload = EXCLUDED.payload
                """,
                rows,
            )

    def add_documents(self, documents: list[dict[str, Any]]) -> int:
        """Insert/update a batch of main-corpus chunks."""
        with self._connect() as connection:
            self._upsert(connection, documents, self.source_type)
        return len(documents)

    def replace_document_chunks(
        self,
        document_id: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Replace all uploaded keyword chunks for one document."""
        with self._connect() as connection:
            deleted = connection.execute(
                f"DELETE FROM {self.TABLE} WHERE source_type = %s AND document_id = %s",
                (self.source_type, document_id),
            ).rowcount
            self._upsert(connection, documents, self.source_type)
        return {
            "added_chunks": len(documents),
            "removed_chunks": int(deleted or 0),
        }

    def delete_documents_by_document_id(self, document_id: str) -> int:
        """Delete indexed chunks for one document."""
        with self._connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM {self.TABLE} WHERE source_type = %s AND document_id = %s",
                (self.source_type, document_id),
            )
            return int(cursor.rowcount or 0)

    def count_documents(self) -> int:
        """Count indexed chunks for this source type."""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM {self.TABLE} WHERE source_type = %s",
                (self.source_type,),
            ).fetchone()
        return int(row["total"] if row else 0)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Search this source type using PostgreSQL full-text ranking."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query.strip():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, payload,
                       ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS score
                FROM {self.TABLE}
                WHERE source_type = %s
                  AND search_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC, chunk_id ASC
                LIMIT %s
                """,
                (query, self.source_type, query, top_k),
            ).fetchall()

        query_terms = set(tokenize(query))
        results = []
        for rank, row in enumerate(rows, start=1):
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            searchable = f"{payload.get('title', '')} {payload.get('text', '')}"
            searchable_terms = set(tokenize(searchable))
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "chunk": payload,
                    "score": float(row["score"] or 0.0),
                    "rank": rank,
                    "matched_terms": sorted(query_terms & searchable_terms),
                }
            )
        return results

    def close(self) -> None:
        """Connections are scoped to each operation; no persistent handle to close."""
        return None
