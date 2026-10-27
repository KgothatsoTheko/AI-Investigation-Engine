
import json
from pathlib import Path

from datasets import load_dataset

from src.ingestion.normalize import (
    example_to_documents,
    example_to_evaluation_item,
)


DATASET_URL = (
    "https://huggingface.co/datasets/"
    "xanhho/2WikiMultihopQA/resolve/main/train.parquet"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading three examples from the dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": DATASET_URL},
        split="train",
        streaming=True,
    )

    examples = list(dataset.take(3))

    documents = []
    evaluation_items = []

    for example in examples:
        documents.extend(
            example_to_documents(example, split="train")
        )

        evaluation_items.append(
            example_to_evaluation_item(example)
        )

    documents_path = OUTPUT_DIR / "sample_passages.jsonl"
    evaluation_path = OUTPUT_DIR / "sample_evaluation.jsonl"

    write_jsonl(documents_path, documents)
    write_jsonl(evaluation_path, evaluation_items)

    unique_ids = {
        document["document_id"]
        for document in documents
    }

    print("\nExport completed.")
    print(f"Examples processed: {len(examples)}")
    print(f"Passage records: {len(documents)}")
    print(f"Unique passage IDs: {len(unique_ids)}")
    print(f"Evaluation records: {len(evaluation_items)}")
    print(f"\nPassages saved to: {documents_path}")
    print(f"Evaluation saved to: {evaluation_path}")


if __name__ == "__main__":
    main()
