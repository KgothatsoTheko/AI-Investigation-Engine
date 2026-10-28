
import json
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"
QDRANT_PATH = DATA_DIR / "qdrant_sample"

SAMPLE_SIZE = 500
CANDIDATE_LIMIT = 20
FINAL_LIMIT = 5


def load_sample(path: Path) -> list[dict]:
    """Load the same initial chunk sample used by Qdrant."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            documents.append(json.loads(line))

            if len(documents) >= SAMPLE_SIZE:
                break

    if not documents:
        raise ValueError("No chunks found.")

    return documents


def display_results(
    heading: str,
    results: list[dict],
    method: str,
) -> None:
    """Print a readable comparison of the ranked results."""
    print(f"\n{'=' * 70}")
    print(heading)
    print("=" * 70)

    if not results:
        print("No matching passages.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nRank: {rank}")
        print(f"Title: {chunk.get('title', 'Unknown')}")

        if method == "semantic":
            print(
                f"Vector similarity: {result['score']:.4f}"
            )

        elif method == "keyword":
            print(f"BM25 score: {result['score']:.4f}")
            print(
                "Matched query terms:",
                ", ".join(result["matched_terms"]),
            )

        elif method == "hybrid":
            print(
                f"RRF score: {result['hybrid_score']:.6f}"
            )
            print("Semantic rank:", result["semantic_rank"])
            print("Keyword rank:", result["keyword_rank"])
            print(
                "Matched query terms:",
                ", ".join(result["matched_terms"]),
            )

        print("Chunk ID:", chunk.get("chunk_id"))
        print("Text:", chunk.get("text", "")[:350])


def main() -> None:
    chunks = load_sample(CHUNK_PATH)

    print(f"Loaded {len(chunks)} chunks.")
    query = input(
        "Enter a question to investigate: "
    ).strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    # Build the local keyword index for the prototype sample.
    keyword_index = BM25Index(chunks)

    keyword_results = keyword_index.search(
        query,
        top_k=CANDIDATE_LIMIT,
    )

    encoder = TextEncoder()
    store = QdrantVectorStore(QDRANT_PATH)

    try:
        query_vector = encoder.encode_query(query)

        vector_points = store.search(
            query_vector,
            limit=CANDIDATE_LIMIT,
        )

        semantic_results = []

        for rank, point in enumerate(vector_points, start=1):
            payload = point.payload or {}

            semantic_results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "chunk": payload,
                    "score": float(point.score),
                    "rank": rank,
                }
            )

        hybrid_results = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            limit=FINAL_LIMIT,
        )

        display_results(
            "SEMANTIC SEARCH",
            semantic_results[:FINAL_LIMIT],
            "semantic",
        )

        display_results(
            "KEYWORD SEARCH — BM25",
            keyword_results[:FINAL_LIMIT],
            "keyword",
        )

        display_results(
            "HYBRID SEARCH — RRF",
            hybrid_results,
            "hybrid",
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()
