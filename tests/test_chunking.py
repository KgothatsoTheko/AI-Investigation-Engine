
import pytest

from src.ingestion.chunking import chunk_document


def make_document():
    return {
        "document_id": "doc-123",
        "title": "Example passage",
        "text": (
            "Sentence one. Sentence two. Sentence three."
        ),
        "sentences": [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
        ],
        "metadata": {"dataset": "test"},
        "provenance": [
            {
                "example_id": "example-1",
                "passage_index": 0,
                "supporting_sentence_indices": [1],
            }
        ],
    }


def test_fixed_window_creates_overlapping_chunks():
    chunks = chunk_document(
        make_document(),
        strategy="fixed_window",
        chunk_size_words=4,
        overlap_words=1,
    )

    assert len(chunks) == 2
    assert chunks[0]["chunking"]["word_count"] == 4
    assert chunks[0]["text"] == "Sentence one. Sentence two."
    assert chunks[0]["source_sentence_indices"] == [0, 1]
    assert chunks[1]["source_sentence_indices"] == [1, 2]


def test_sentence_grouping_preserves_sentence_boundaries():
    chunks = chunk_document(
        make_document(),
        strategy="sentence_grouped",
        chunk_size_words=4,
        overlap_sentences=1,
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == "Sentence one. Sentence two."
    assert chunks[1]["text"] == "Sentence two. Sentence three."

    assert chunks[0]["source_sentence_indices"] == [0, 1]
    assert chunks[1]["source_sentence_indices"] == [1, 2]


def test_chunks_keep_source_metadata():
    chunks = chunk_document(
        make_document(),
        strategy="sentence_grouped",
        chunk_size_words=4,
    )

    assert chunks[0]["document_id"] == "doc-123"
    assert chunks[0]["title"] == "Example passage"
    assert chunks[0]["provenance"][0]["example_id"] == "example-1"


def test_chunk_ids_are_deterministic():
    first = chunk_document(
        make_document(),
        strategy="sentence_grouped",
        chunk_size_words=4,
    )
    second = chunk_document(
        make_document(),
        strategy="sentence_grouped",
        chunk_size_words=4,
    )

    assert [c["chunk_id"] for c in first] == [
        c["chunk_id"] for c in second
    ]


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(
            make_document(),
            strategy="unknown",
        )


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_document(
            make_document(),
            strategy="fixed_window",
            chunk_size_words=4,
            overlap_words=4,
        )
