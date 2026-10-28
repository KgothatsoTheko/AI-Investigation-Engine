
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "qdrant_full"
)


def main() -> None:
    query = input(
        "Enter a question to search the full corpus: "
    ).strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    encoder = TextEncoder()

    store = QdrantVectorStore(
        database_path=QDRANT_PATH,
        collection_name="investigation_chunks",
    )

    try:
        count = store.count_points()
        print(f"\nIndexed chunks available: {count:,}")

        query_vector = encoder.encode_query(query)

        results = store.search(
            query_vector,
            limit=5,
        )

        print("\nTop matching passages:\n")

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            print("=" * 70)
            print(f"Rank: {rank}")
            print(f"Similarity: {result.score:.4f}")
            print(f"Title: {payload.get('title', 'Unknown')}")
            print(f"Chunk ID: {payload.get('chunk_id')}")
            print(f"Source document: {payload.get('document_id')}")
            print(
                "Sentence indices:",
                payload.get("source_sentence_indices", []),
            )
            print("\nPassage:")
            print(payload.get("text", "")[:700])
            print()

    finally:
        store.close()


if __name__ == "__main__":
    main()
