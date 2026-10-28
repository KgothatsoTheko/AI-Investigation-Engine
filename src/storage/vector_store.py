
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

        # Persistent local database: data remains after closing
        # the client and running the script again.
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

    def recreate_collection(self, vector_size: int) -> None:
        """Create a fresh collection with the required vector size."""
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
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
            payload = dict(chunk)

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk["chunk_id"]),
                    vector=embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding),
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
            query=query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector),
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

    def close(self) -> None:
        """Release the database client."""
        self.client.close()
