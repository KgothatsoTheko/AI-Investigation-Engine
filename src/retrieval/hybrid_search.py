
from typing import Any


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    limit: int = 5,
    rrf_k: int = 60,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Combine semantic and keyword rankings.

    Each result contributes weight / (rrf_k + rank).
    Ranks begin at 1, and raw scores from the two systems
    are not added together.
    """
    if limit <= 0:
        raise ValueError("limit must be positive.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    if semantic_weight < 0 or keyword_weight < 0:
        raise ValueError("Weights cannot be negative.")

    fused: dict[str, dict[str, Any]] = {}

    result_sources = [
        ("semantic", semantic_results, semantic_weight),
        ("keyword", keyword_results, keyword_weight),
    ]

    for source_name, results, weight in result_sources:
        for rank, result in enumerate(results, start=1):
            chunk = result.get("chunk", {})
            chunk_id = (
                result.get("chunk_id")
                or chunk.get("chunk_id")
            )

            if not chunk_id:
                raise ValueError(
                    f"{source_name} result has no chunk_id."
                )

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                    "hybrid_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_score": None,
                    "keyword_score": None,
                    "matched_terms": [],
                }

            combined = fused[chunk_id]

            # A result found by both methods accumulates two
            # rank-based contributions.
            combined["hybrid_score"] += (
                weight / (rrf_k + rank)
            )

            combined[f"{source_name}_rank"] = rank
            combined[f"{source_name}_score"] = result.get(
                "score"
            )

            if source_name == "keyword":
                combined["matched_terms"] = result.get(
                    "matched_terms", []
                )

    return sorted(
        fused.values(),
        key=lambda result: (
            -result["hybrid_score"],
            result["chunk_id"],
        ),
    )[:limit]
