
import json
import sqlite3
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value consistently for comparisons."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


class SQLiteCorpusStore:
    """
    Store unique source passages in SQLite.

    SQLite provides disk-based storage and a primary key that
    prevents duplicate document IDs.
    """

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.database_path)
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    def add_document(self, document: dict[str, Any]) -> bool:
        """
        Insert a new document or merge provenance into an existing one.

        Returns True if a new unique document was inserted.
        Returns False if the document ID already existed.
        """
        document_id = document.get("document_id")
        title = document.get("title")
        text = document.get("text")

        if not document_id or not title or not text:
            raise ValueError(
                "A document must have an ID, title and text."
            )

        metadata_json = canonical_json(
            document.get("metadata", {})
        )

        incoming_provenance = document.get("provenance", [])
        provenance_json = canonical_json(incoming_provenance)

        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO documents (
                document_id,
                title,
                text,
                metadata_json,
                provenance_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                title,
                text,
                metadata_json,
                provenance_json,
            ),
        )

        # A new ID was inserted successfully.
        if cursor.rowcount == 1:
            return True

        # The ID already exists. Merge any new provenance.
        row = self.connection.execute(
            """
            SELECT provenance_json
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                f"Could not retrieve document {document_id}."
            )

        existing_provenance = json.loads(row[0])

        seen = {
            canonical_json(item)
            for item in existing_provenance
        }

        changed = False

        for item in incoming_provenance:
            item_key = canonical_json(item)

            if item_key not in seen:
                existing_provenance.append(item)
                seen.add(item_key)
                changed = True

        if changed:
            self.connection.execute(
                """
                UPDATE documents
                SET provenance_json = ?
                WHERE document_id = ?
                """,
                (
                    canonical_json(existing_provenance),
                    document_id,
                ),
            )

        return False

    def count_documents(self) -> int:
        """Return the number of distinct stored document IDs."""
        row = self.connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()

        return int(row[0])

    def commit(self) -> None:
        """Persist the current database transaction."""
        self.connection.commit()

    def export_jsonl(self, output_path: Path | str) -> None:
        """Export unique source passages to JSON Lines format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection.commit()

        rows = self.connection.execute(
            """
            SELECT
                document_id,
                title,
                text,
                metadata_json,
                provenance_json
            FROM documents
            ORDER BY document_id
            """
        )

        with output_path.open("w", encoding="utf-8") as file:
            for (
                document_id,
                title,
                text,
                metadata_json,
                provenance_json,
            ) in rows:
                document = {
                    "document_id": document_id,
                    "title": title,
                    "text": text,
                    "metadata": json.loads(metadata_json),
                    "provenance": json.loads(provenance_json),
                }

                file.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
