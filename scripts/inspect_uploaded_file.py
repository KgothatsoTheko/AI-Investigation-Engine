
import argparse
import json

from src.ingestion.file_loader import load_document


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an uploaded document."
    )

    parser.add_argument(
        "file_path",
        help="Path to a TXT, PDF or DOCX file.",
    )

    args = parser.parse_args()

    try:
        document = load_document(args.file_path)

    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("Document loaded successfully.")
    print(f"Document ID: {document['document_id']}")
    print(f"Title: {document['title']}")
    print(f"Format: {document['metadata']['file_format']}")
    print(f"Characters: {len(document['text'])}")
    print(f"Sentences: {len(document['sentences'])}")

    print("\nMetadata:")
    print(
        json.dumps(
            document["metadata"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nText preview:")
    print(document["text"][:1000])


if __name__ == "__main__":
    main()
