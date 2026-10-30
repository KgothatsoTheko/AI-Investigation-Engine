import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "uploaded_documents.sqlite"


class SQLiteUploadStore:
    """Uploaded-document store using SQLite locally or hosted PostgreSQL in deployment.

    The historical class name is retained so existing callers and tests remain compatible.
    Set SUPABASE_DB_URL (recommended) or DATABASE_URL to enable PostgreSQL.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        database_url: str | None = None,
    ):
        self.database_url = (
            database_url
            or os.getenv("SUPABASE_DB_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )

        if self.database_url:
            self.db_path = None
        else:
            self.db_path = Path(db_path or DEFAULT_DB_PATH)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator:
        if self.database_url:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "Hosted PostgreSQL storage requires psycopg. "
                    "Install requirements.txt."
                ) from exc

            with psycopg.connect(
                self.database_url,
                row_factory=dict_row,
            ) as connection:
                yield connection
            return

        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def _placeholder(self) -> str:
        return "%s" if self.database_url else "?"

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    sentences_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _serialize(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    def save_document(self, document: dict) -> str:
        """Insert a document or update the existing document with its ID."""
        for field in ("document_id", "title", "text"):
            if not document.get(field):
                raise ValueError(
                    f"Document is missing required field: {field}"
                )

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        placeholder = self._placeholder

        with self._connection() as connection:
            connection.execute(
                f"""
                INSERT INTO uploaded_documents (
                    document_id,
                    title,
                    text,
                    metadata_json,
                    sentences_json,
                    provenance_json,
                    created_at,
                    updated_at
                ) VALUES ({', '.join([placeholder] * 8)})
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    text = excluded.text,
                    metadata_json = excluded.metadata_json,
                    sentences_json = excluded.sentences_json,
                    provenance_json = excluded.provenance_json,
                    updated_at = excluded.updated_at
                """,
                (
                    document["document_id"],
                    document["title"],
                    document["text"],
                    self._serialize(document.get("metadata", {})),
                    self._serialize(document.get("sentences", [])),
                    self._serialize(document.get("provenance", [])),
                    now,
                    now,
                ),
            )

        return document["document_id"]

    def get_document(self, document_id: str) -> dict | None:
        """Return a saved document, or None when it does not exist."""
        placeholder = self._placeholder

        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM uploaded_documents WHERE document_id = {placeholder}",
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "document_id": row["document_id"],
            "title": row["title"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"]),
            "sentences": json.loads(row["sentences_json"]),
            "provenance": json.loads(row["provenance_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_documents(self) -> list[dict]:
        """Return document summaries, newest updated first."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    title,
                    metadata_json,
                    created_at,
                    updated_at
                FROM uploaded_documents
                ORDER BY updated_at DESC, document_id ASC
                """
            ).fetchall()

        return [
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def count(self) -> int:
        """Return the total number of saved documents."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS document_count FROM uploaded_documents"
            ).fetchone()

        return int(row["document_count"])

    def delete_document(self, document_id: str) -> bool:
        """Delete a saved document. Return False when it does not exist."""
        placeholder = self._placeholder

        with self._connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM uploaded_documents WHERE document_id = {placeholder}",
                (document_id,),
            )
            return cursor.rowcount > 0
