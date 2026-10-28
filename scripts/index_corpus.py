
import json
from pathlib import Path
from collections.abc import Iterator

from src.embeddings.encoder import TextEncoder
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"
QDRANT_PATH = DATA_DIR / "qdrant_full"

INDEX_BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 32
PROGRESS_INTERVAL = 25


def read_batches(
    path: Path,
    batch_size: int,
) -> Iterator[list[dict]]:
    """Read JSONL chunks from disk without loading all of them."""
    batch = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main() -> None:
    if not CHUNK_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNK_PATH}\n"
            "Run python -m scripts.compare_chunking first."
        )

    print("Starting full-corpus embedding and indexing.")
    print(f"Chunk source: {CHUNK_PATH}")
    print(f"Qdrant database: {QDRANT_PATH}")
    print(f"Index batch size: {INDEX_BATCH_SIZE}")

    encoder = TextEncoder()

    store = QdrantVectorStore(
        database_path=QDRANT_PATH,
        collection_name="investigation_chunks",
    )

    total_indexed = 0
    total_batches = 0

    try:
        # This is a fresh full-corpus build. It replaces the
        # collection in qdrant_full but leaves qdrant_sample alone.
        store.recreate_collection(
            vector_size=encoder.dimension
        )

        for total_batches, chunks in enumerate(
            read_batches(CHUNK_PATH, INDEX_BATCH_SIZE),
            start=1,
        ):
            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = encoder.encode_documents(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )

            store.upsert_batch(
                chunks,
                embeddings,
            )

            total_indexed += len(chunks)

            if total_batches % PROGRESS_INTERVAL == 0:
                print(
                    f"Batches processed: {total_batches:,} | "
                    f"Chunks indexed: {total_indexed:,}"
                )

        actual_count = store.count_points()

        print("\nFull-corpus indexing completed.")
        print(f"Embedding model: all-MiniLM-L6-v2")
        print(f"Vector dimensions: {encoder.dimension}")
        print(f"Chunks indexed: {total_indexed:,}")
        print(f"Vectors stored: {actual_count:,}")
        print(f"Database location: {QDRANT_PATH}")

        if actual_count != total_indexed:
            raise RuntimeError(
                "The number of stored vectors does not match "
                "the number of indexed chunks."
            )

    finally:
        store.close()


if __name__ == "__main__":
    main()
