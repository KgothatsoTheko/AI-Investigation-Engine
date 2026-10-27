
import json
from pathlib import Path
from typing import Any

from src.ingestion.chunking import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"

FIXED_OUTPUT = DATA_DIR / "chunks_fixed_window.jsonl"
SENTENCE_OUTPUT = DATA_DIR / "chunks_sentence_grouped.jsonl"

CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 20
OVERLAP_SENTENCES = 1


def create_stats() -> dict[str, Any]:
    return {
        "source_documents": 0,
        "documents_split": 0,
        "total_chunks": 0,
        "total_words": 0,
        "min_words": None,
        "max_words": 0,
        "oversized_chunks": 0,
        "empty_chunks": 0,
    }


def record_chunks(
    stats: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> None:
    if len(chunks) > 1:
        stats["documents_split"] += 1

    for chunk in chunks:
        word_count = chunk["chunking"]["word_count"]

        stats["total_chunks"] += 1
        stats["total_words"] += word_count

        if word_count == 0:
            stats["empty_chunks"] += 1

        current_min = stats["min_words"]

        if current_min is None or word_count < current_min:
            stats["min_words"] = word_count

        stats["max_words"] = max(
            stats["max_words"],
            word_count,
        )

        if word_count > CHUNK_SIZE_WORDS:
            stats["oversized_chunks"] += 1


def write_chunks(
    file,
    chunks: list[dict[str, Any]],
) -> None:
    for chunk in chunks:
        file.write(
            json.dumps(chunk, ensure_ascii=False) + "\n"
        )


def print_stats(
    strategy_name: str,
    stats: dict[str, Any],
) -> None:
    total_chunks = stats["total_chunks"]
    average_words = (
        stats["total_words"] / total_chunks
        if total_chunks
        else 0
    )

    print(f"\n{strategy_name}")
    print("-" * len(strategy_name))
    print(f"Source documents: {stats['source_documents']:,}")
    print(f"Documents split into multiple chunks: "
          f"{stats['documents_split']:,}")
    print(f"Total chunks: {total_chunks:,}")
    print(f"Average words per chunk: {average_words:.2f}")
    print(f"Minimum words in a chunk: {stats['min_words']}")
    print(f"Maximum words in a chunk: {stats['max_words']}")
    print(f"Chunks over the word target: "
          f"{stats['oversized_chunks']:,}")
    print(f"Empty chunks: {stats['empty_chunks']:,}")


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\n"
            "Run python -m scripts.build_corpus first."
        )

    fixed_stats = create_stats()
    sentence_stats = create_stats()

    print("Comparing chunking strategies...")
    print(f"Input corpus: {CORPUS_PATH}")

    with (
        CORPUS_PATH.open("r", encoding="utf-8") as source_file,
        FIXED_OUTPUT.open("w", encoding="utf-8") as fixed_file,
        SENTENCE_OUTPUT.open("w", encoding="utf-8") as sentence_file,
    ):
        for line in source_file:
            if not line.strip():
                continue

            document = json.loads(line)

            fixed_stats["source_documents"] += 1
            sentence_stats["source_documents"] += 1

            fixed_chunks = chunk_document(
                document,
                strategy="fixed_window",
                chunk_size_words=CHUNK_SIZE_WORDS,
                overlap_words=OVERLAP_WORDS,
            )

            sentence_chunks = chunk_document(
                document,
                strategy="sentence_grouped",
                chunk_size_words=CHUNK_SIZE_WORDS,
                overlap_sentences=OVERLAP_SENTENCES,
            )

            write_chunks(fixed_file, fixed_chunks)
            write_chunks(sentence_file, sentence_chunks)

            record_chunks(fixed_stats, fixed_chunks)
            record_chunks(sentence_stats, sentence_chunks)

    print("\nChunking comparison completed.")

    print_stats("Strategy A: Fixed-size sliding window", fixed_stats)
    print_stats("Strategy B: Sentence-aware grouping", sentence_stats)

    print("\nOutput files:")
    print(FIXED_OUTPUT)
    print(SENTENCE_OUTPUT)


if __name__ == "__main__":
    main()
