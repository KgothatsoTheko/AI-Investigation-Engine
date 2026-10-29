import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploaded_documents.sqlite"
)


class SQLiteUploadStore:
    """Persistent storage for user-uploaded documents."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
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
        required_fields = ("document_id", "title", "text")

        for field in required_fields:
            if not document.get(field):
                raise ValueError(f"Document is missing required field: {field}")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO uploaded_documents (
                    document_id,
                    title,
                    text,
                    metadata_json,
                    sentences_json,
                    provenance_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM uploaded_documents
                WHERE document_id = ?
                """,
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

    def delete_document(self, document_id: str) -> bool:
        """Delete a saved document. Return False when it does not exist."""
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM uploaded_documents
                WHERE document_id = ?
                """,
                (document_id,),
            )

            return cursor.rowcount > 0

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM uploaded_documents"
            ).fetchone()

        return row["total"]