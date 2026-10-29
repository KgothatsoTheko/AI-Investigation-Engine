
import re
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.retrieval.hybrid_search import (reciprocal_rank_fusion_multiple)
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.storage.vector_store import QdrantVectorStore


def normalize_evidence_text(text: str) -> str:
    """Normalize text for duplicate-result detection."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


class HybridEvidenceRetriever:
    """Retrieve and rank evidence from both full-corpus indexes."""

    def __init__(
        self,
        data_dir: Path | str,
        candidate_limit: int = 25,
        final_limit: int = 12,
    ):
        if candidate_limit <= 0 or final_limit <= 0:
            raise ValueError("Search limits must be positive.")

        data_dir = Path(data_dir)

        keyword_path = data_dir / "keyword_index.sqlite"
        qdrant_path = data_dir / "qdrant_full"

        if not keyword_path.exists():
            raise FileNotFoundError(
                f"Keyword index not found: {keyword_path}"
            )

        if not qdrant_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: {qdrant_path}"
            )

        self.candidate_limit = candidate_limit
        self.final_limit = final_limit

        self.encoder = TextEncoder()

        self.keyword_index = SQLiteBM25Index(
            keyword_path
        )

        self.vector_store = QdrantVectorStore(
            database_path=qdrant_path,
            collection_name="investigation_chunks",
        )

        # Uploaded-document indexes are optional.
        # The original full-corpus indexes remain unchanged.
        upload_keyword_path = (
            data_dir / "upload_keyword_index.sqlite"
        )
        upload_vector_path = data_dir / "qdrant_uploads"

        self.upload_keyword_index = None
        self.upload_vector_store = None

        if upload_keyword_path.exists():
            self.upload_keyword_index = SQLiteBM25Index(
                upload_keyword_path
            )

        if upload_vector_path.exists():
            self.upload_vector_store = QdrantVectorStore(
                database_path=upload_vector_path,
                collection_name="uploaded_chunks",
            )

    def retrieve(
            self,
            question: str,
        ) -> list[dict[str, Any]]:
            """Retrieve and fuse evidence from the main and upload indexes."""
            if not question.strip():
                raise ValueError("The question cannot be empty.")

            # 1. Search the main keyword index.
            main_keyword_results = self.keyword_index.search(
                question,
                top_k=self.candidate_limit,
            )

            # 2. Search the main semantic index.
            query_vector = self.encoder.encode_query(question)

            main_vector_points = self.vector_store.search(
                query_vector,
                limit=self.candidate_limit,
            )

            main_semantic_results = [
                {
                    "chunk_id": (point.payload or {}).get("chunk_id"),
                    "chunk": point.payload or {},
                    "score": float(point.score),
                }
                for point in main_vector_points
            ]

            # The four lists will be fused into one ranking.
            ranked_lists = [
                (
                    "main_semantic",
                    "semantic",
                    main_semantic_results,
                    1.0,
                ),
                (
                    "main_keyword",
                    "keyword",
                    main_keyword_results,
                    1.0,
                ),
            ]

            # 3. Search the uploaded-document keyword index if available.
            if self.upload_keyword_index is not None:
                upload_keyword_results = (
                    self.upload_keyword_index.search(
                        question,
                        top_k=self.candidate_limit,
                    )
                )

                ranked_lists.append(
                    (
                        "upload_keyword",
                        "keyword",
                        upload_keyword_results,
                        1.0,
                    )
                )

            # 4. Search the uploaded-document vector index if its
            # collection exists. An empty upload index is valid.
            if (
                self.upload_vector_store is not None
                and self.upload_vector_store.has_collection()
            ):
                upload_vector_points = self.upload_vector_store.search(
                    query_vector,
                    limit=self.candidate_limit,
                )

                upload_semantic_results = [
                    {
                        "chunk_id": (point.payload or {}).get("chunk_id"),
                        "chunk": point.payload or {},
                        "score": float(point.score),
                    }
                    for point in upload_vector_points
                ]

                ranked_lists.append(
                    (
                        "upload_semantic",
                        "semantic",
                        upload_semantic_results,
                        1.0,
                    )
                )

            # 5. Fuse rankings across all available indexes.
            fused_results = reciprocal_rank_fusion_multiple(
                ranked_lists=ranked_lists,
                limit=self.candidate_limit,
            )

            # 6. Deduplicate repeated evidence.
            selected = []
            seen_content = set()

            for result in fused_results:
                chunk = result.get("chunk", {})

                title_key = normalize_evidence_text(
                    chunk.get("title", "")
                )

                text_key = normalize_evidence_text(
                    chunk.get("text", "")
                )

                if not text_key:
                    continue

                identity = (title_key, text_key)

                if identity in seen_content:
                    continue

                seen_content.add(identity)
                selected.append(result)

                if len(selected) >= self.final_limit:
                    break

            return selected

    def close(self) -> None:
        """Release all open index resources."""
        try:
            self.keyword_index.close()
        finally:
            try:
                self.vector_store.close()
            finally:
                if self.upload_keyword_index is not None:
                    self.upload_keyword_index.close()

                if self.upload_vector_store is not None:
                    self.upload_vector_store.close()