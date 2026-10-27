
import json
from pathlib import Path

from datasets import load_dataset

from src.ingestion.corpus_store import SQLiteCorpusStore
from src.ingestion.normalize import (
    example_to_documents,
    example_to_evaluation_item,
)


DATASET_REVISION = (
    "6ef4eb1bc94ca4a768bb86809fafe777ad795995"
)

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "xanhho/2WikiMultihopQA/resolve/"
    f"{DATASET_REVISION}/train.parquet"
)

TARGET_UNIQUE_DOCUMENTS = 50_000
EVALUATION_SEED_SIZE = 100
PROGRESS_INTERVAL = 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DATABASE_PATH = OUTPUT_DIR / "corpus_staging.sqlite"
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"
EVALUATION_PATH = OUTPUT_DIR / "evaluation_seed.jsonl"


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def remove_old_staging_database() -> None:
    """
    Start each build from a clean database.

    SQLite may also have WAL and shared-memory files.
    """
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{DATABASE_PATH}{suffix}")

        if path.exists():
            path.unlink()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remove_old_staging_database()

    store = SQLiteCorpusStore(DATABASE_PATH)

    examples_scanned = 0
    source_passages_seen = 0
    unique_passages_inserted = 0
    duplicate_passages_seen = 0
    evaluation_seed = []

    print("Starting reproducible corpus build.")
    print(f"Dataset revision: {DATASET_REVISION}")
    print(
        "Target unique passage records: "
        f"{TARGET_UNIQUE_DOCUMENTS:,}"
    )

    try:
        dataset = load_dataset(
            "parquet",
            data_files={"train": DATASET_URL},
            split="train",
            streaming=True,
        )

        for example in dataset:
            examples_scanned += 1

            # Keep evaluation data separate from source passages.
            if len(evaluation_seed) < EVALUATION_SEED_SIZE:
                evaluation_seed.append(
                    example_to_evaluation_item(example)
                )

            documents = example_to_documents(
                example,
                split="train",
            )

            for document in documents:
                source_passages_seen += 1

                was_inserted = store.add_document(document)

                if was_inserted:
                    unique_passages_inserted += 1
                else:
                    duplicate_passages_seen += 1

            # Commit periodically rather than after every passage.
            if examples_scanned % PROGRESS_INTERVAL == 0:
                store.commit()

                print(
                    f"Examples: {examples_scanned:,} | "
                    f"Unique passages: "
                    f"{unique_passages_inserted:,}/"
                    f"{TARGET_UNIQUE_DOCUMENTS:,} | "
                    f"Duplicates: {duplicate_passages_seen:,}"
                )

            # Finish the current example before stopping.
            if (
                unique_passages_inserted
                >= TARGET_UNIQUE_DOCUMENTS
            ):
                break

        store.commit()

        final_count = store.count_documents()

        if final_count < TARGET_UNIQUE_DOCUMENTS:
            raise RuntimeError(
                "The dataset ended before the target was reached. "
                f"Only {final_count:,} unique passages were stored."
            )

        store.export_jsonl(CORPUS_PATH)

    finally:
        store.close()

    write_jsonl(EVALUATION_PATH, evaluation_seed)

    print("\nCorpus build completed.")
    print(f"Examples scanned: {examples_scanned:,}")
    print(f"Source passages seen: {source_passages_seen:,}")
    print(f"Unique passage records: {final_count:,}")
    print(f"Duplicate encounters: {duplicate_passages_seen:,}")
    print(f"Evaluation seed questions: {len(evaluation_seed):,}")
    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Evaluation seed: {EVALUATION_PATH}")
    print(f"Staging database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
