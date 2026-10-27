# Script to inspect the 2WikiMultihopQA dataset
import json

from datasets import load_dataset


DATASET_URL = (
    "https://huggingface.co/datasets/"
    "xanhho/2WikiMultihopQA/resolve/main/train.parquet"
)


def parse_json_field(value):
    """Decode a JSON string if the dataset stores a field as text."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def inspect_context(raw_context):
    """Print a short summary of the context passages."""
    contexts = parse_json_field(raw_context)

    if not isinstance(contexts, list):
        print("Context format:", type(contexts).__name__)
        print("Context preview:", str(contexts)[:300])
        return

    print("Number of context passages:", len(contexts))

    for index, item in enumerate(contexts[:2], start=1):
        if isinstance(item, dict):
            title = item.get("title", "Unknown title")
            content = item.get(
                "content",
                item.get("sentences", []),
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, content = item[0], item[1]
        else:
            print(f"Passage {index}: unexpected format")
            continue

        if isinstance(content, list):
            preview = " ".join(str(sentence) for sentence in content[:2])
            sentence_count = len(content)
        else:
            preview = str(content)
            sentence_count = "unknown"

        print(f"\nPassage {index}")
        print("Title:", title)
        print("Sentence count:", sentence_count)
        print("Preview:", preview[:300])


def main():
    print("Loading a small sample from 2WikiMultihopQA...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": DATASET_URL},
        split="train",
        streaming=True,
    )

    # Read only the first three examples.
    for index, example in enumerate(dataset.take(3), start=1):
        print("\n" + "=" * 60)
        print(f"EXAMPLE {index}")
        print("=" * 60)

        print("Available fields:", list(example.keys()))
        print("ID:", example.get("_id", example.get("id")))
        print("Question:", example.get("question"))
        print("Answer:", example.get("answer"))
        print("Question type:", example.get("type"))

        print("\nCONTEXT")
        inspect_context(example.get("context", []))

        supporting_facts = parse_json_field(
            example.get("supporting_facts", [])
        )

        print("\nSUPPORTING FACTS")
        print(str(supporting_facts)[:500])

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
