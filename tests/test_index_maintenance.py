from src.storage.vector_store import QdrantVectorStore
from src.retrieval.sqlite_bm25 import SQLiteBM25Index


def make_chunk(chunk_id, document_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": f"Title {document_id}",
        "text": text,
        "metadata": {},
        "provenance": [],
    }


def test_qdrant_ensure_collection_preserves_existing_data(tmp_path):
    store = QdrantVectorStore(
        tmp_path / "vectors",
        collection_name="test_chunks",
    )

    try:
        store.ensure_collection(vector_size=3)

        chunk = make_chunk(
            "chunk_a",
            "document_a",
            "vector preservation test",
        )

        store.upsert_batch([chunk], [[1.0, 0.0, 0.0]])

        store.ensure_collection(vector_size=3)

        assert store.has_collection()
        assert store.count_points() == 1

    finally:
        store.close()


def test_qdrant_rejects_incompatible_vector_dimension(tmp_path):
    store = QdrantVectorStore(
        tmp_path / "vectors",
        collection_name="test_chunks",
    )

    try:
        store.ensure_collection(vector_size=3)

        try:
            store.ensure_collection(vector_size=4)
        except ValueError as exc:
            assert "dimension mismatch" in str(exc).lower()
        else:
            raise AssertionError("Expected a dimension mismatch.")

    finally:
        store.close()


def test_qdrant_deletes_obsolete_chunks_without_affecting_other_documents(
    tmp_path,
):
    store = QdrantVectorStore(
        tmp_path / "vectors",
        collection_name="test_chunks",
    )

    try:
        store.ensure_collection(vector_size=3)

        chunks = [
            make_chunk("a1", "doc_a", "current evidence"),
            make_chunk("a2", "doc_a", "obsolete evidence"),
            make_chunk("b1", "doc_b", "unrelated evidence"),
        ]

        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        store.upsert_batch(chunks, embeddings)

        removed = store.delete_document_except_chunks(
            "doc_a",
            keep_chunk_ids={"a1"},
        )

        assert removed == 1
        assert store.count_points() == 2

        removed = store.delete_document("doc_a")

        assert removed == 1
        assert store.count_points() == 1

    finally:
        store.close()


def test_qdrant_delete_missing_collection_is_safe(tmp_path):
    store = QdrantVectorStore(
        tmp_path / "vectors",
        collection_name="not_created",
    )

    try:
        assert store.delete_document("missing_document") == 0
    finally:
        store.close()


def test_bm25_deletion_removes_stale_terms_and_preserves_other_documents(
    tmp_path,
):
    index = SQLiteBM25Index(tmp_path / "keyword.sqlite")

    try:
        index.add_documents([
            make_chunk(
                "a1",
                "doc_a",
                "uniqueoldterm evidence",
            ),
            make_chunk(
                "b1",
                "doc_b",
                "unrelated surviving evidence",
            ),
        ])

        removed = index.delete_documents_by_document_id("doc_a")

        assert removed == 1
        assert index.count_documents() == 1
        assert index.search("uniqueoldterm", top_k=5) == []
        assert len(index.search("surviving", top_k=5)) == 1

    finally:
        index.close()


def test_bm25_replacement_removes_old_chunks(tmp_path):
    index = SQLiteBM25Index(tmp_path / "keyword.sqlite")

    try:
        index.add_documents([
            make_chunk(
                "old_chunk",
                "doc_a",
                "obsoletebluephrase",
            ),
        ])

        result = index.replace_document_chunks(
            document_id="doc_a",
            documents=[
                make_chunk(
                    "new_chunk",
                    "doc_a",
                    "replacementgreenphrase",
                ),
            ],
        )

        assert result == {
            "removed_chunks": 1,
            "added_chunks": 1,
        }

        assert index.count_documents() == 1
        assert index.search("obsoletebluephrase", top_k=5) == []
        assert len(index.search("replacementgreenphrase", top_k=5)) == 1

    finally:
        index.close()