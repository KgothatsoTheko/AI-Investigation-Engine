"""Copy the local main Qdrant collection to Qdrant Cloud.

Run from the project root after setting QDRANT_URL and QDRANT_API_KEY in the
current terminal. Stop the local Streamlit app first so the local database is free.
"""
import argparse
import os
from pathlib import Path

from qdrant_client import QdrantClient, models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing qdrant_full (default: data/processed)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true", help="Delete the remote collection before copying")
    args = parser.parse_args()

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url:
        raise SystemExit("Missing QDRANT_URL in the current terminal environment.")
    local_path = args.data_dir / "qdrant_full"
    if not local_path.exists():
        raise SystemExit(f"Local vector database not found: {local_path}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    source = QdrantClient(path=str(local_path))
    destination = QdrantClient(url=url, api_key=api_key or None, timeout=120)
    collection_name = "investigation_chunks"

    try:
        if not source.collection_exists(collection_name):
            raise SystemExit(f"Local collection '{collection_name}' was not found.")
        source_info = source.get_collection(collection_name)
        vector_config = source_info.config.params.vectors
        vector_size = getattr(vector_config, "size", None)
        if not vector_size:
            raise SystemExit("Expected a single-vector collection with a known vector size.")
        source_count = int(source.count(collection_name, exact=True).count)
        print(f"Source collection: {source_count:,} points, vector size {vector_size}.")

        remote_exists = destination.collection_exists(collection_name)
        if args.recreate and remote_exists:
            print("Deleting existing remote collection as requested...")
            destination.delete_collection(collection_name)
            remote_exists = False
        if not remote_exists:
            destination.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            remote_info = destination.get_collection(collection_name)
            remote_size = getattr(remote_info.config.params.vectors, "size", None)
            if remote_size != vector_size:
                raise SystemExit(
                    f"Remote collection dimension {remote_size} differs from local {vector_size}. "
                    "Create a new remote collection or use --recreate only if replacing it is intended."
                )

        offset = None
        copied = 0
        while True:
            records, next_offset = source.scroll(
                collection_name=collection_name,
                limit=args.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if records:
                points = [
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload or {},
                    )
                    for record in records
                ]
                destination.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                copied += len(points)
                print(f"Copied {copied:,} / {source_count:,} points", end="\r", flush=True)
            if next_offset is None:
                break
            offset = next_offset
        final_count = int(destination.count(collection_name, exact=True).count)
        print(f"\nRemote collection now contains {final_count:,} points.")
        if final_count < source_count:
            raise SystemExit(
                f"Verification failed: expected at least {source_count:,} remote points."
            )
        print("Qdrant migration completed and count verification passed.")
    finally:
        source.close()
        destination.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
