from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.ingestion.chunking import chunk_document
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
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

        # Reuse the saved document library.
        self.document_store = document_store or SQLiteUploadStore()

        # Load the embedding model only when indexing is requested.
        # Tests can inject a lightweight fake encoder.
        self.encoder = encoder

        # These are separate from qdrant_full and keyword_index.sqlite.
        self.vector_store = QdrantVectorStore(
            database_path=self.data_dir / "qdrant_uploads",
            collection_name="uploaded_chunks",
        )

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
            raise KeyError(
                f"Saved document not found: {document_id}"
            )

        encoder = self._get_encoder()

        chunks = chunk_document(
            document,
            strategy="sentence_grouped",
            chunk_size_words=100,
            overlap_sentences=1,
        )

        # Preserve the existing index if the embedding dimension
        # is incompatible with the configured collection.
        self.vector_store.ensure_collection(
            vector_size=encoder.dimension
        )

        texts = [chunk["text"] for chunk in chunks]

        embeddings = encoder.encode_documents(
            texts,
            batch_size=32,
            show_progress_bar=False,
        )

        if len(chunks) != len(embeddings):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # Upsert first. Existing IDs are overwritten; new IDs are added.
        # This means a retry can repair an interrupted indexing attempt.
        self.vector_store.upsert_batch(chunks, embeddings)

        # Replace the document's complete keyword index entries.
        keyword_result = self.keyword_index.replace_document_chunks(
            document_id=document_id,
            documents=chunks,
        )

        # Remove obsolete vectors only after the current chunks have
        # been upserted and the keyword replacement has succeeded.
        current_chunk_ids = {
            chunk["chunk_id"] for chunk in chunks
        }

        removed_vectors = (
            self.vector_store.delete_document_except_chunks(
                document_id=document_id,
                keep_chunk_ids=current_chunk_ids,
            )
        )

        return {
            "document_id": document_id,
            "title": document["title"],
            "chunking_strategy": "sentence_grouped",
            "chunks_indexed": len(chunks),
            "vectors_upserted": len(embeddings),
            "keyword_chunks_added": keyword_result["added_chunks"],
            "previous_keyword_chunks_removed": (
                keyword_result["removed_chunks"]
            ),
            "obsolete_vectors_removed": removed_vectors,
        }

    def delete_indexed_document(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        """Remove a document's chunks from both upload indexes."""

        removed_keyword_chunks = (
            self.keyword_index.delete_documents_by_document_id(
                document_id
            )
        )

        removed_vectors = self.vector_store.delete_document(
            document_id
        )

        return {
            "document_id": document_id,
            "keyword_chunks_removed": removed_keyword_chunks,
            "vectors_removed": removed_vectors,
        }

    def index_all_documents(self) -> list[dict[str, Any]]:
        """Index all documents currently in the saved library."""

        document_summaries = self.document_store.list_documents()
        results = []

        for item in document_summaries:
            results.append(
                self.index_document(item["document_id"])
            )

        return results

    def close(self) -> None:
        """Release index resources."""

        try:
            self.keyword_index.close()
        finally:
            self.vector_store.close()