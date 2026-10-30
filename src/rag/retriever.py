import os
import re
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.retrieval.hybrid_search import reciprocal_rank_fusion_multiple
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.retrieval.postgres_keyword import PostgresKeywordIndex
from src.storage.vector_store import QdrantVectorStore


def normalize_evidence_text(text: str) -> str:
    """Normalize text for duplicate-result detection."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


class HybridEvidenceRetriever:
    """Retrieve and rank evidence from all available vector and keyword indexes."""

    def __init__(
        self,
        data_dir: Path | str,
        candidate_limit: int = 25,
        final_limit: int = 12,
        encoder: TextEncoder | None = None,
    ):
        if candidate_limit <= 0 or final_limit <= 0:
            raise ValueError("Search limits must be positive.")

        data_dir = Path(data_dir)
        self.candidate_limit = candidate_limit
        self.final_limit = final_limit
        self.encoder = encoder
        database_url = (
            os.getenv("SUPABASE_DB_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        remote_qdrant = bool(os.getenv("QDRANT_URL", "").strip())

        keyword_path = data_dir / "keyword_index.sqlite"
        qdrant_path = data_dir / "qdrant_full"
        upload_keyword_path = data_dir / "upload_keyword_index.sqlite"
        upload_vector_path = data_dir / "qdrant_uploads"

        self.keyword_index = None
        self.upload_keyword_index = None
        self.vector_store = None
        self.upload_vector_store = None

        if database_url:
            self.keyword_index = PostgresKeywordIndex(
                database_url=database_url,
                source_type="main",
            )
            self.upload_keyword_index = PostgresKeywordIndex(
                database_url=database_url,
                source_type="uploaded",
            )
        else:
            if keyword_path.exists():
                self.keyword_index = SQLiteBM25Index(keyword_path)
            if upload_keyword_path.exists():
                self.upload_keyword_index = SQLiteBM25Index(upload_keyword_path)

        # If Qdrant Cloud is configured, the local directory is not required.
        if remote_qdrant or qdrant_path.exists():
            candidate_store = QdrantVectorStore(
                database_path=qdrant_path,
                collection_name="investigation_chunks",
            )
            if candidate_store.has_collection():
                self.vector_store = candidate_store
            else:
                candidate_store.close()

        if remote_qdrant or upload_vector_path.exists():
            candidate_store = QdrantVectorStore(
                database_path=upload_vector_path,
                collection_name="uploaded_chunks",
            )
            if candidate_store.has_collection():
                self.upload_vector_store = candidate_store
            else:
                candidate_store.close()

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve and fuse evidence from available indexes."""
        if not question.strip():
            raise ValueError("The question cannot be empty.")

        ranked_lists = []
        for name, index in (
            ("main_keyword", self.keyword_index),
            ("upload_keyword", self.upload_keyword_index),
        ):
            if index is None:
                continue
            results = index.search(question, top_k=self.candidate_limit)
            ranked_lists.append((name, "keyword", results, 1.0))

        vector_stores = [
            (name, store)
            for name, store in (
                ("main_semantic", self.vector_store),
                ("upload_semantic", self.upload_vector_store),
            )
            if store is not None
        ]
        if vector_stores:
            if self.encoder is None:
                self.encoder = TextEncoder()
            query_vector = self.encoder.encode_query(question)
            for name, store in vector_stores:
                points = store.search(query_vector, limit=self.candidate_limit)
                results = [
                    {
                        "chunk_id": (point.payload or {}).get("chunk_id"),
                        "chunk": point.payload or {},
                        "score": float(point.score),
                    }
                    for point in points
                ]
                ranked_lists.append((name, "semantic", results, 1.0))

        if not ranked_lists:
            return []

        fused_results = reciprocal_rank_fusion_multiple(
            ranked_lists=ranked_lists,
            limit=self.candidate_limit,
        )
        selected = []
        seen_content = set()
        for result in fused_results:
            chunk = result.get("chunk", {})
            title_key = normalize_evidence_text(chunk.get("title", ""))
            text_key = normalize_evidence_text(chunk.get("text", ""))
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
        """Release all available index resources."""
        for resource in (
            self.keyword_index,
            self.vector_store,
            self.upload_keyword_index,
            self.upload_vector_store,
        ):
            if resource is not None:
                resource.close()
