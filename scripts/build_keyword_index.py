
import json
from pathlib import Path

from src.retrieval.sqlite_bm25 import SQLiteBM25Index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"
DATABASE_PATH = DATA_DIR / "keyword_index.sqlite"

BATCH_SIZE = 256


def read_batches(path: Path, batch_size: int):
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


def remove_old_database(path: Path) -> None:
    """Remove the SQLite database and its WAL sidecar files."""
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")

        if candidate.exists():
            candidate.unlink()


def main() -> None:
    if not CHUNK_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNK_PATH}"
        )

    remove_old_database(DATABASE_PATH)

    index = SQLiteBM25Index(
        DATABASE_PATH,
        reset=True,
    )

    total_documents = 0
    total_batches = 0

    try:
        for total_batches, batch in enumerate(
            read_batches(CHUNK_PATH, BATCH_SIZE),
            start=1,
        ):
            index.add_documents(batch)
            total_documents += len(batch)

            if total_batches % 25 == 0:
                print(
                    f"Batches: {total_batches:,} | "
                    f"Indexed chunks: {total_documents:,}"
                )

        stored_count = index.count_documents()

        print("\nKeyword index completed.")
        print(f"Input chunks: {total_documents:,}")
        print(f"Stored chunks: {stored_count:,}")
        print(f"Database: {DATABASE_PATH}")

        if stored_count != total_documents:
            raise RuntimeError(
                "Keyword index count does not match input count."
            )

    finally:
        index.close()


if __name__ == "__main__":
    main()
