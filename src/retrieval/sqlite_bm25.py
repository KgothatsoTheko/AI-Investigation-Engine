
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.retrieval.bm25 import tokenize


class SQLiteBM25Index:
    """Persistent BM25 keyword index backed by SQLite."""

    def __init__(
        self,
        database_path: Path | str,
        reset: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.k1 = k1
        self.b = b

        self.connection = sqlite3.connect(
            str(self.database_path)
        )

        self.connection.execute(
            "PRAGMA journal_mode=WAL"
        )
        self.connection.execute(
            "PRAGMA synchronous=NORMAL"
        )

        if reset:
            self.connection.executescript(
                """
                DROP TABLE IF EXISTS postings;
                DROP TABLE IF EXISTS terms;
                DROP TABLE IF EXISTS documents;
                """
            )

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                chunk_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                document_length INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS terms (
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            );

            CREATE INDEX IF NOT EXISTS idx_postings_chunk
                ON postings(chunk_id);
            """
        )
        self.connection.commit()

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """
        Add a batch of chunks and update document frequencies.

        This operation is intended for a new index build.
        """
        document_rows = []
        posting_rows = []
        batch_document_frequencies = Counter()

        seen_chunk_ids = set()

        for document in documents:
            chunk_id = document.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Every chunk must have a chunk_id."
                )

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID in batch: {chunk_id}"
                )
            seen_chunk_ids.add(chunk_id)

            searchable_text = (
                f"{document.get('title', '')} "
                f"{document.get('text', '')}"
            )

            frequencies = Counter(
                tokenize(searchable_text)
            )

            document_length = sum(frequencies.values())

            document_rows.append(
                (
                    chunk_id,
                    json.dumps(
                        document,
                        ensure_ascii=False,
                    ),
                    document_length,
                )
            )

            for term, frequency in frequencies.items():
                posting_rows.append(
                    (term, chunk_id, frequency)
                )

            # Each distinct term contributes once per document.
            batch_document_frequencies.update(frequencies.keys())

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO documents (
                    chunk_id,
                    payload_json,
                    document_length
                )
                VALUES (?, ?, ?)
                """,
                document_rows,
            )

            self.connection.executemany(
                """
                INSERT INTO postings (
                    term,
                    chunk_id,
                    term_frequency
                )
                VALUES (?, ?, ?)
                """,
                posting_rows,
            )

            self.connection.executemany(
                """
                INSERT INTO terms (
                    term,
                    document_frequency
                )
                VALUES (?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    document_frequency =
                        document_frequency + excluded.document_frequency
                """,
                list(batch_document_frequencies.items()),
            )

    def count_documents(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()

        return int(row[0])

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Run custom BM25 scoring against the persistent index."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = list(dict.fromkeys(tokenize(query)))

        if not query_terms:
            return []

        stats = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(AVG(document_length), 0)
            FROM documents
            """
        ).fetchone()

        document_count = int(stats[0])
        average_length = float(stats[1])

        if document_count == 0 or average_length == 0:
            return []

        scores = defaultdict(float)
        matched_terms = defaultdict(set)

        for term in query_terms:
            term_row = self.connection.execute(
                """
                SELECT document_frequency
                FROM terms
                WHERE term = ?
                """,
                (term,),
            ).fetchone()

            if term_row is None:
                continue

            document_frequency = int(term_row[0])

            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - document_frequency
                    + 0.5
                )
                / (document_frequency + 0.5)
            )

            postings = self.connection.execute(
                """
                SELECT
                    p.chunk_id,
                    p.term_frequency,
                    d.document_length
                FROM postings AS p
                JOIN documents AS d
                    ON d.chunk_id = p.chunk_id
                WHERE p.term = ?
                """,
                (term,),
            )

            for chunk_id, term_frequency, length in postings:
                length_adjustment = self.k1 * (
                    1
                    - self.b
                    + self.b * length / average_length
                )

                score = (
                    inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + length_adjustment)
                )

                scores[chunk_id] += score
                matched_terms[chunk_id].add(term)

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        results = []

        for rank, (chunk_id, score) in enumerate(
            ranked,
            start=1,
        ):
            row = self.connection.execute(
                """
                SELECT payload_json
                FROM documents
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

            if row is None:
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": json.loads(row[0]),
                    "score": float(score),
                    "rank": rank,
                    "matched_terms": sorted(
                        matched_terms[chunk_id]
                    ),
                }
            )

        return results

    def close(self) -> None:
        self.connection.close()
