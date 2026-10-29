import pytest

from src.ingestion.upload_indexer import UploadIndexer
from src.storage.upload_store import SQLiteUploadStore


class FakeEncoder:
    """Predictable encoder for tests; does not load a real ML model."""

    dimension = 3

    def encode_documents(
        self,
        texts,
        batch_size=32,
        show_progress_bar=False,
    ):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


def make_document(document_id, text):
    sentences = [
        sentence.strip()
        for sentence in text.split(". ")
        if sentence.strip()
    ]

    return {
        "document_id": document_id,
        "title": f"Document {document_id}",
        "text": text,
        "sentences": sentences,
        "metadata": {
            "source_type": "uploaded_file",
            "file_format": "txt",
            "original_filename": f"{document_id}.txt",
        },
        "provenance": [
            {
                "source_type": "uploaded_file",
                "original_filename": f"{document_id}.txt",
            }
        ],
    }


@pytest.fixture
def indexing_environment(tmp_path):
    document_store = SQLiteUploadStore(
        tmp_path / "uploaded_documents.sqlite"
    )

    indexer = UploadIndexer(
        data_dir=tmp_path / "indexes",
        document_store=document_store,
        encoder=FakeEncoder(),
    )

    yield document_store, indexer

    indexer.close()


def test_index_document_adds_chunks_to_both_indexes(
    indexing_environment,
):
    store, indexer = indexing_environment

    document = make_document(
        "upload_doc_a",
        "The uniquezebraphrase describes the system. "
        "Another sentence provides context.",
    )

    store.save_document(document)

    result = indexer.index_document(document["document_id"])

    assert result["chunks_indexed"] > 0
    assert result["vectors_upserted"] == result["chunks_indexed"]

    assert (
        indexer.keyword_index.count_documents()
        == result["chunks_indexed"]
    )

    assert (
        indexer.vector_store.count_points()
        == result["chunks_indexed"]
    )

    results = indexer.keyword_index.search(
        "uniquezebraphrase",
        top_k=5,
    )

    assert results
    assert results[0]["chunk"]["document_id"] == "upload_doc_a"


def test_reindexing_same_document_does_not_duplicate_chunks(
    indexing_environment,
):
    store, indexer = indexing_environment

    document = make_document(
        "upload_doc_a",
        "The uniqueotterphrase is part of the evidence.",
    )

    store.save_document(document)

    first = indexer.index_document("upload_doc_a")
    second = indexer.index_document("upload_doc_a")

    assert first["chunks_indexed"] == second["chunks_indexed"]
    assert (
        indexer.keyword_index.count_documents()
        == second["chunks_indexed"]
    )
    assert (
        indexer.vector_store.count_points()
        == second["chunks_indexed"]
    )


def test_reindexing_updated_document_removes_old_content(
    indexing_environment,
):
    store, indexer = indexing_environment

    store.save_document(
        make_document(
            "upload_doc_a",
            "The obsoletepurplephrase is in the old version.",
        )
    )

    indexer.index_document("upload_doc_a")

    store.save_document(
        make_document(
            "upload_doc_a",
            "The replacementgoldphrase is in the new version.",
        )
    )

    result = indexer.index_document("upload_doc_a")

    assert result["chunks_indexed"] > 0

    assert indexer.keyword_index.search(
        "obsoletepurplephrase",
        top_k=5,
    ) == []

    new_results = indexer.keyword_index.search(
        "replacementgoldphrase",
        top_k=5,
    )

    assert new_results
    assert indexer.keyword_index.count_documents() == (
        result["chunks_indexed"]
    )
    assert indexer.vector_store.count_points() == (
        result["chunks_indexed"]
    )


def test_deleting_indexed_document_preserves_other_documents(
    indexing_environment,
):
    store, indexer = indexing_environment

    store.save_document(
        make_document(
            "upload_doc_a",
            "The deletedcopperphrase belongs to document A.",
        )
    )

    store.save_document(
        make_document(
            "upload_doc_b",
            "The survivingtealphrase belongs to document B.",
        )
    )

    indexer.index_document("upload_doc_a")
    result_b = indexer.index_document("upload_doc_b")

    deletion = indexer.delete_indexed_document("upload_doc_a")

    assert deletion["keyword_chunks_removed"] > 0
    assert deletion["vectors_removed"] > 0

    assert indexer.keyword_index.search(
        "deletedcopperphrase",
        top_k=5,
    ) == []

    remaining = indexer.keyword_index.search(
        "survivingtealphrase",
        top_k=5,
    )

    assert remaining
    assert remaining[0]["chunk"]["document_id"] == "upload_doc_b"

    assert indexer.keyword_index.count_documents() == (
        result_b["chunks_indexed"]
    )

    assert indexer.vector_store.count_points() == (
        result_b["chunks_indexed"]
    )


def test_indexing_missing_saved_document_raises_error(
    indexing_environment,
):
    _, indexer = indexing_environment

    with pytest.raises(KeyError, match="Saved document not found"):
        indexer.index_document("does_not_exist")