
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

QDRANT_PATH = DATA_DIR / "qdrant_full"
KEYWORD_DB_PATH = DATA_DIR / "keyword_index.sqlite"

CANDIDATE_LIMIT = 20
FINAL_LIMIT = 5


def display_results(results: list[dict]) -> None:
    print("\nHYBRID SEARCH RESULTS")
    print("=" * 70)

    if not results:
        print("No matching passages found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nRank: {rank}")
        print(f"RRF score: {result['hybrid_score']:.6f}")
        print(f"Title: {chunk.get('title', 'Unknown')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Source document: {chunk.get('document_id')}")
        print(f"Semantic rank: {result['semantic_rank']}")
        print(f"Keyword rank: {result['keyword_rank']}")
        print("Matched terms:", ", ".join(result["matched_terms"]))
        print(
            "Sentence indices:",
            chunk.get("source_sentence_indices", []),
        )
        print("\nPassage:")
        print(chunk.get("text", "")[:700])


def main() -> None:
    if not KEYWORD_DB_PATH.exists():
        raise FileNotFoundError(
            "Keyword index not found. Run "
            "python -m scripts.build_keyword_index first."
        )

    query = input(
        "Enter a question to investigate: "
    ).strip()

    if not query:
        print("Please enter a non-empty query.")
        return

    encoder = TextEncoder()

    keyword_index = SQLiteBM25Index(
        KEYWORD_DB_PATH
    )
    vector_store = QdrantVectorStore(
        database_path=QDRANT_PATH,
        collection_name="investigation_chunks",
    )

    try:
        keyword_count = keyword_index.count_documents()
        vector_count = vector_store.count_points()

        print(f"\nKeyword-index chunks: {keyword_count:,}")
        print(f"Vector-index chunks: {vector_count:,}")

        if keyword_count != vector_count:
            print(
                "Warning: the keyword and vector indexes have "
                "different record counts. Check their build outputs."
            )

        # Keyword retrieval over the full indexed corpus.
        keyword_results = keyword_index.search(
            query,
            top_k=CANDIDATE_LIMIT,
        )

        # Semantic retrieval over the full vector index.
        query_vector = encoder.encode_query(query)

        vector_points = vector_store.search(
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

        display_results(hybrid_results)

    finally:
        keyword_index.close()
        vector_store.close()


if __name__ == "__main__":
    main()
