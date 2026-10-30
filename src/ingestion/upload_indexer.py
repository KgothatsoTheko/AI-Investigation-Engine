import os
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.ingestion.chunking import chunk_document
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.retrieval.postgres_keyword import PostgresKeywordIndex
from src.storage.upload_store import SQLiteUploadStore
from src.storage.vector_store import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class UploadIndexer:
    """Index, update, and remove chunks from uploaded documents."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        document_store: SQLiteUploadStore | None = None,
        encoder: Any | None = None,
    ):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.document_store = document_store or SQLiteUploadStore()
        self.encoder = encoder

        self.vector_store = QdrantVectorStore(
            database_path=self.data_dir / "qdrant_uploads",
            collection_name="uploaded_chunks",
        )
        database_url = (
            os.getenv("SUPABASE_DB_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        if database_url:
            self.keyword_index = PostgresKeywordIndex(
                database_url=database_url,
                source_type="uploaded",
            )
        else:
            self.keyword_index = SQLiteBM25Index(
                database_path=self.data_dir / "upload_keyword_index.sqlite",
            )

    def _get_encoder(self) -> TextEncoder:
        if self.encoder is None:
            self.encoder = TextEncoder()
        return self.encoder

    def index_document(self, document_id: str) -> dict[str, Any]:
        """Index a saved document, replacing its previous chunks."""
        document = self.document_store.get_document(document_id)
        if document is None:
            raise KeyError(f"Saved document not found: {document_id}")

        encoder = self._get_encoder()
        chunks = chunk_document(
            document,
            strategy="sentence_grouped",
            chunk_size_words=100,
            overlap_sentences=1,
        )
        self.vector_store.ensure_collection(vector_size=encoder.dimension)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = encoder.encode_documents(
            texts,
            batch_size=32,
            show_progress_bar=False,
        )
        if len(chunks) != len(embeddings):
            raise RuntimeError("Embedding count does not match chunk count.")

        self.vector_store.upsert_batch(chunks, embeddings)
        keyword_result = self.keyword_index.replace_document_chunks(
            document_id=document_id,
            documents=chunks,
        )
        current_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        removed_vectors = self.vector_store.delete_document_except_chunks(
            document_id=document_id,
            keep_chunk_ids=current_chunk_ids,
        )
        return {
            "document_id": document_id,
            "title": document["title"],
            "chunking_strategy": "sentence_grouped",
            "chunks_indexed": len(chunks),
            "vectors_upserted": len(embeddings),
            "keyword_chunks_added": keyword_result["added_chunks"],
            "previous_keyword_chunks_removed": keyword_result["removed_chunks"],
            "obsolete_vectors_removed": removed_vectors,
        }

    def delete_indexed_document(self, document_id: str) -> dict[str, Any]:
        """Remove a document's chunks from both upload indexes."""
        removed_keyword_chunks = self.keyword_index.delete_documents_by_document_id(
            document_id
        )
        removed_vectors = self.vector_store.delete_document(document_id)
        return {
            "document_id": document_id,
            "keyword_chunks_removed": removed_keyword_chunks,
            "vectors_removed": removed_vectors,
        }

    def delete_saved_document(self, document_id: str) -> dict[str, Any]:
        """Remove indexed chunks and then delete the saved original."""
        document = self.document_store.get_document(document_id)
        if document is None:
            raise KeyError(f"Saved document not found: {document_id}")
        index_result = self.delete_indexed_document(document_id)
        deleted = self.document_store.delete_document(document_id)
        if not deleted:
            raise RuntimeError(f"Could not delete saved document: {document_id}")
        return {
            **index_result,
            "title": document["title"],
            "saved_document_deleted": True,
        }

    def index_all_documents(self) -> list[dict[str, Any]]:
        """Index all documents currently in the saved library."""
        return [
            self.index_document(item["document_id"])
            for item in self.document_store.list_documents()
        ]

    def close(self) -> None:
        """Release index resources."""
        try:
            self.keyword_index.close()
        finally:
            self.vector_store.close()
