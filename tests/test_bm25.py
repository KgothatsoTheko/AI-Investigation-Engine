
from src.retrieval.bm25 import BM25Index, tokenize


def make_documents():
    return [
        {
            "chunk_id": "apple",
            "title": "Apple",
            "text": "Apples are crunchy fruits.",
        },
        {
            "chunk_id": "orange",
            "title": "Orange",
            "text": "Oranges are citrus fruits.",
        },
        {
            "chunk_id": "film",
            "title": "Film directors",
            "text": "A director makes films.",
        },
    ]


def test_bm25_ranks_matching_terms():
    index = BM25Index(make_documents())

    results = index.search(
        "apple crunchy fruit",
        top_k=3,
    )

    assert results[0]["chunk_id"] == "apple"
    assert "apple" in results[0]["matched_terms"]


def test_bm25_returns_no_results_for_unknown_terms():
    index = BM25Index(make_documents())

    results = index.search(
        "xyz_nonexistent_term",
        top_k=5,
    )

    assert results == []


def test_bm25_rejects_invalid_limit():
    index = BM25Index(make_documents())

    try:
        index.search("apple", top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for top_k=0."
        )

def test_tokenize_removes_stopwords_but_preserves_negation():
    assert tokenize("Which films did Mario Bava direct?") == [
        "films",
        "mario",
        "bava",
        "direct",
    ]

    assert tokenize("The film is not directed by Mario.") == [
        "film",
        "not",
        "directed",
        "mario",
    ]