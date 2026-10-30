import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_DIR = (PROJECT_ROOT / "data" / "processed").resolve()


class QdrantVectorStore:
    """Qdrant vector store with safe separation between app and test databases.

    The application's persistent indexes under PROJECT_DATA_DIR use Qdrant
    Cloud when QDRANT_URL is configured. Temporary/custom paths (including
    pytest tmp_path directories) always use local Qdrant unless use_remote=True
    is explicitly supplied.

    This prevents tests that call recreate_collection() from deleting or
    modifying production collections.
    """

    def __init__(
        self,
        database_path: Path | str,
        collection_name: str = "investigation_chunks",
        use_remote: bool | None = None,
    ):
        database_path = Path(database_path).expanduser().resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        configured_url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None

        if use_remote is None:
            # Only the app's normal persistent data directory is remote by
            # default. pytest tmp_path and other custom paths remain local.
            try:
                database_path.relative_to(PROJECT_DATA_DIR)
                use_remote = True
            except ValueError:
                use_remote = False

        self.remote_url = configured_url if use_remote else ""

        if self.remote_url:
            if not api_key:
                raise RuntimeError(
                    "QDRANT_URL is configured for remote storage, but "
                    "QDRANT_API_KEY is missing."
                )
            self.client = QdrantClient(
                url=self.remote_url,
                api_key=api_key,
                timeout=60,
            )
        else:
            self.client = QdrantClient(path=str(database_path))

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

    def _ensure_document_id_index(self) -> None:
        """Create the keyword payload index required for document filters."""
        if not self.remote_url or not self.has_collection():
            return

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def recreate_collection(self, vector_size: int) -> None:
        """Delete and recreate this collection; intended for explicit resets."""
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
        self._ensure_document_id_index()

    def ensure_collection(self, vector_size: int) -> None:
        """Create a collection if missing, validate dimensions, and index filters."""
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
            self._ensure_document_id_index()
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
                f"Vector dimension mismatch: collection uses {existing_size}, "
                f"requested {vector_size}."
            )

        self._ensure_document_id_index()

    def upsert_batch(
        self,
        chunks: list[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update a batch of embedded chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError("Every chunk must have a chunk_id.")

            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector,
                    payload=dict(chunk),
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

        vector = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector)
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
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
        return int(result.count)

    def delete_document_except_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        """Delete obsolete vectors for a document while preserving current chunks."""
        if not document_id:
            raise ValueError("document_id cannot be empty.")

        if not self.has_collection():
            return 0

        self._ensure_document_id_index()

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
                if payload.get("chunk_id") not in keep_chunk_ids:
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
        """Release the Qdrant client."""
        self.client.close()
