
import re
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.retrieval.hybrid_search import reciprocal_rank_fusion
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

    def retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """Return diversified, fused search results."""
        if not question.strip():
            raise ValueError("The question cannot be empty.")

        keyword_results = self.keyword_index.search(
            question,
            top_k=self.candidate_limit,
        )

        query_vector = self.encoder.encode_query(question)

        vector_points = self.vector_store.search(
            query_vector,
            limit=self.candidate_limit,
        )

        semantic_results = []

        for point in vector_points:
            payload = point.payload or {}

            semantic_results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "chunk": payload,
                    "score": float(point.score),
                }
            )

        fused_results = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            limit=self.candidate_limit,
        )

        # Avoid giving the generator several differently formatted
        # copies of the same title and passage.
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
        """Release database resources."""
        self.keyword_index.close()
        self.vector_store.close()
