"""Import a JSONL chunk file into the persistent PostgreSQL keyword index.

Example:
  python scripts/migrate_chunks_to_postgres.py --input data/processed/YOUR_SENTENCE_CHUNKS.jsonl

Set SUPABASE_DB_URL in the current terminal first. The input file should contain
one JSON object per line with at least chunk_id and text fields.
"""
import argparse
import json
import os
from pathlib import Path

from src.retrieval.postgres_keyword import PostgresKeywordIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Sentence-grouped chunk JSONL file")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--source-type", choices=["main"], default="main")
    args = parser.parse_args()

    database_url = (
        os.getenv("SUPABASE_DB_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not database_url:
        raise SystemExit("Missing SUPABASE_DB_URL (or DATABASE_URL) in the current terminal environment.")
    if not args.input.exists():
        raise SystemExit(f"Chunk file not found: {args.input}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    index = PostgresKeywordIndex(database_url=database_url, source_type="main")
    total = 0
    batch = []
    try:
        with args.input.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
                batch.append(record)
                if len(batch) >= args.batch_size:
                    index.add_documents(batch)
                    total += len(batch)
                    batch.clear()
                    print(f"Imported {total:,} chunks", end="\r", flush=True)
            if batch:
                index.add_documents(batch)
                total += len(batch)
        final_count = index.count_documents()
        print(f"\nImported {total:,} records. Main keyword index now has {final_count:,} records.")
        if final_count < total:
            raise SystemExit("Verification warning: database count is below imported row count.")
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
