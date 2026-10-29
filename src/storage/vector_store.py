import uuid
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    """Store and search embedded text chunks in Qdrant."""

    def __init__(
        self,
        database_path: Path | str,
        collection_name: str = "investigation_chunks",
    ):
        database_path = Path(database_path)
        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name

        # Persistent local database.
        self.client = QdrantClient(
            path=str(database_path)
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert a chunk ID into a deterministic UUID."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"ai-investigation-engine:{chunk_id}",
            )
        )

    def has_collection(self) -> bool:
        """Check whether the configured collection exists."""
        return self.client.collection_exists(
            collection_name=self.collection_name
        )

    def recreate_collection(self, vector_size: int) -> None:
        """Create a fresh collection with the required vector size."""
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")

        if self.has_collection():
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def ensure_collection(self, vector_size: int) -> None:
        """
        Create the collection if missing.

        Existing collections are preserved, and their vector
        dimensions must match the requested embedding dimension.
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")

        if not self.has_collection():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(
            collection_name=self.collection_name
        )
        existing_vectors = info.config.params.vectors
        existing_size = getattr(existing_vectors, "size", None)

        if existing_size is None:
            raise ValueError(
                "The existing collection uses an unsupported "
                "or named-vector configuration."
            )

        if existing_size != vector_size:
            raise ValueError(
                f"Vector dimension mismatch: collection uses "
                f"{existing_size}, requested {vector_size}."
            )

    def upsert_batch(
        self,
        chunks: list[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update a batch of embedded chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number "
                "of embeddings."
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Every chunk must have a chunk_id."
                )

            payload = dict(chunk)

            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
    ):
        """Retrieve the closest chunks to a query vector."""
        if limit <= 0:
            raise ValueError("limit must be positive.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=(
                query_vector.tolist()
                if hasattr(query_vector, "tolist")
                else list(query_vector)
            ),
            limit=limit,
            with_payload=True,
        )

        return response.points

    def count_points(self) -> int:
        """Return the number of indexed vectors."""
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def delete_document_except_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        """
        Delete obsolete vectors belonging to a document.

        Chunks in keep_chunk_ids are preserved. This method is
        intended to run after replacement chunks are upserted.
        """
        if not document_id:
            raise ValueError("document_id cannot be empty.")

        if not self.has_collection():
            return 0

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

        offset = None
        point_ids_to_delete = []

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")

                if chunk_id not in keep_chunk_ids:
                    point_ids_to_delete.append(point.id)

            if next_offset is None:
                break

            offset = next_offset

        if point_ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids_to_delete
                ),
                wait=True,
            )

        return len(point_ids_to_delete)

    def delete_document(self, document_id: str) -> int:
        """Delete all vectors belonging to one document."""
        return self.delete_document_except_chunks(
            document_id=document_id,
            keep_chunk_ids=set(),
        )

    def close(self) -> None:
        """Release the database client."""
        self.client.close()