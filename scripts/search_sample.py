
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = PROJECT_ROOT / "data" / "processed" / "qdrant_sample"


def main() -> None:
    query = input(
        "Enter a topic or question to search for: "
    ).strip()

    if not query:
        print("Please provide a non-empty query.")
        return

    encoder = TextEncoder()
    store = QdrantVectorStore(QDRANT_PATH)

    try:
        query_vector = encoder.encode_query(query)

        results = store.search(
            query_vector,
            limit=5,
        )

        print(f"\nTop {len(results)} matching passages:\n")

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            print("=" * 70)
            print(f"Rank: {rank}")
            print(f"Similarity score: {result.score:.4f}")
            print(f"Title: {payload.get('title', 'Unknown')}")
            print(f"Chunk ID: {payload.get('chunk_id')}")
            print(f"Source document: {payload.get('document_id')}")

            sentence_indices = payload.get(
                "source_sentence_indices",
                [],
            )

            print(
                "Source sentence indices:",
                sentence_indices,
            )

            print("\nPassage:")
            print(payload.get("text", "")[:700])
            print()

    finally:
        store.close()


if __name__ == "__main__":
    main()
