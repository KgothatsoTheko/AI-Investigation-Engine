
from src.retrieval.hybrid_search import reciprocal_rank_fusion


def test_rrf_promotes_a_result_found_by_both_methods():
    semantic_results = [
        {
            "chunk_id": "a",
            "chunk": {"chunk_id": "a", "text": "Passage A"},
            "score": 0.90,
        },
        {
            "chunk_id": "b",
            "chunk": {"chunk_id": "b", "text": "Passage B"},
            "score": 0.80,
        },
    ]

    keyword_results = [
        {
            "chunk_id": "b",
            "chunk": {"chunk_id": "b", "text": "Passage B"},
            "score": 3.50,
            "matched_terms": ["director"],
        },
        {
            "chunk_id": "c",
            "chunk": {"chunk_id": "c", "text": "Passage C"},
            "score": 2.00,
            "matched_terms": ["director"],
        },
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        limit=3,
    )

    assert results[0]["chunk_id"] == "b"
    assert results[0]["semantic_rank"] == 2
    assert results[0]["keyword_rank"] == 1
    assert results[0]["matched_terms"] == ["director"]


def test_rrf_rejects_invalid_limit():
    try:
        reciprocal_rank_fusion([], [], limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for limit=0."
        )
