
import pytest

from src.retrieval.sqlite_bm25 import SQLiteBM25Index


def sample_chunks():
    return [
        {
            "chunk_id": "apple",
            "title": "Apple",
            "text": "Apples are crunchy red fruits.",
        },
        {
            "chunk_id": "orange",
            "title": "Orange",
            "text": "Oranges are juicy citrus fruits.",
        },
        {
            "chunk_id": "film",
            "title": "Film director",
            "text": "A director makes films.",
        },
    ]


def test_sqlite_bm25_ranks_keyword_matches(tmp_path):
    index = SQLiteBM25Index(
        tmp_path / "test.sqlite",
        reset=True,
    )

    try:
        index.add_documents(sample_chunks())

        results = index.search(
            "apple crunchy",
            top_k=3,
        )

        assert results[0]["chunk_id"] == "apple"
        assert results[0]["matched_terms"] == [
            "apple",
            "crunchy",
        ]
        assert index.count_documents() == 3

    finally:
        index.close()


def test_sqlite_bm25_returns_no_unknown_matches(tmp_path):
    index = SQLiteBM25Index(
        tmp_path / "empty_results.sqlite",
        reset=True,
    )

    try:
        index.add_documents(sample_chunks())

        results = index.search(
            "xyz_nonexistent_term",
            top_k=5,
        )

        assert results == []

    finally:
        index.close()


def test_sqlite_bm25_rejects_invalid_limit(tmp_path):
    index = SQLiteBM25Index(
        tmp_path / "invalid_limit.sqlite",
        reset=True,
    )

    try:
        with pytest.raises(
            ValueError,
            match="top_k must be positive",
        ):
            index.search("apple", top_k=0)

    finally:
        index.close()
