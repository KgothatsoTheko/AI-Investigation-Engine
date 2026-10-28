
from src.storage.vector_store import QdrantVectorStore


def test_store_and_retrieve_vectors(tmp_path):
    store = QdrantVectorStore(
        tmp_path / "qdrant_test"
    )

    try:
        store.recreate_collection(vector_size=3)

        chunks = [
            {
                "chunk_id": "chunk-a",
                "document_id": "doc-a",
                "title": "Passage A",
                "text": "The first passage is about apples.",
            },
            {
                "chunk_id": "chunk-b",
                "document_id": "doc-b",
                "title": "Passage B",
                "text": "The second passage is about oranges.",
            },
        ]

        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        store.upsert_batch(chunks, embeddings)

        assert store.count_points() == 2

        results = store.search(
            [1.0, 0.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0].payload["chunk_id"] == "chunk-a"
        assert results[0].score > results[1].score

    finally:
        store.close()


def test_query_limit_must_be_positive(tmp_path):
    store = QdrantVectorStore(
        tmp_path / "qdrant_limit_test"
    )

    try:
        store.recreate_collection(vector_size=3)

        try:
            store.search([1.0, 0.0, 0.0], limit=0)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError for limit=0."
            )

    finally:
        store.close()
