import argparse
import json

from src.ingestion.upload_indexer import UploadIndexer
from src.storage.upload_store import SQLiteUploadStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage indexes for uploaded documents."
    )

    parser.add_argument(
        "document_id",
        nargs="?",
        help="ID of the saved document to index.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Index every document in the saved library.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved documents without loading the embedding model.",
    )

    parser.add_argument(
        "--delete",
        metavar="DOCUMENT_ID",
        help="Remove a document from both upload indexes.",
    )

    args = parser.parse_args()

    if args.all and (args.document_id or args.delete):
        parser.error("--all cannot be combined with another operation.")

    if args.delete and args.document_id:
        parser.error(
            "Provide either a document ID or --delete, not both."
        )

    store = SQLiteUploadStore()

    if args.list:
        for document in store.list_documents():
            print(
                f'{document["document_id"]} | '
                f'{document["title"]}'
            )
        return

    if not args.all and not args.delete and not args.document_id:
        parser.error(
            "Provide a document ID, --all, --delete DOCUMENT_ID, "
            "or --list."
        )

    indexer = UploadIndexer(document_store=store)

    try:
        if args.delete:
            result = indexer.delete_indexed_document(args.delete)
        elif args.all:
            result = indexer.index_all_documents()
        else:
            result = indexer.index_document(args.document_id)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    finally:
        indexer.close()


if __name__ == "__main__":
    main()