
import json
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"
QDRANT_PATH = DATA_DIR / "qdrant_sample"

SAMPLE_SIZE = 500
EMBEDDING_BATCH_SIZE = 32
UPLOAD_BATCH_SIZE = 64


def load_sample(path: Path) -> list[dict]:
    """Read only the first SAMPLE_SIZE chunk records."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}\n"
            "Run python -m scripts.compare_chunking first."
        )

    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunks.append(json.loads(line))

            if len(chunks) >= SAMPLE_SIZE:
                break

    if not chunks:
        raise ValueError("No chunks were found in the input file.")

    return chunks


def main() -> None:
    chunks = load_sample(CHUNK_PATH)

    print(f"Loaded {len(chunks)} chunks.")
    print("Loading the embedding model...")

    encoder = TextEncoder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = encoder.encode_documents(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
    )

    store = QdrantVectorStore(QDRANT_PATH)

    try:
        # This sample collection is rebuilt from scratch each run.
        store.recreate_collection(
            vector_size=encoder.dimension
        )

        print("Uploading vectors to Qdrant...")

        for start in range(0, len(chunks), UPLOAD_BATCH_SIZE):
            end = min(
                start + UPLOAD_BATCH_SIZE,
                len(chunks),
            )

            store.upsert_batch(
                chunks[start:end],
                embeddings[start:end],
            )

            print(
                f"Indexed {end}/{len(chunks)} chunks."
            )

        indexed_count = store.count_points()

        print("\nSample indexing completed.")
        print(f"Model: all-MiniLM-L6-v2")
        print(f"Vector dimensions: {encoder.dimension}")
        print(f"Expected vectors: {len(chunks)}")
        print(f"Stored vectors: {indexed_count}")
        print(f"Database path: {QDRANT_PATH}")

        if indexed_count != len(chunks):
            raise RuntimeError(
                "Stored vector count does not match "
                "the expected count."
            )

    finally:
        store.close()


if __name__ == "__main__":
    main()
