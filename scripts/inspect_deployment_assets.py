from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_ASSETS = [
    "corpus.jsonl",
    "evaluation_seed.jsonl",
    "chunks_sentence_grouped.jsonl",
    "chunks_fixed_window.jsonl",
    "keyword_index.sqlite",
    "qdrant_full",
]

UPLOAD_ASSETS = [
    "uploaded_documents.sqlite",
    "upload_keyword_index.sqlite",
    "qdrant_uploads",
]

EVALUATION_ASSETS = [
    "evaluation_answer_results.jsonl",
    "single_vs_multistep_results.jsonl",
]


def get_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    if path.is_dir():
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )

    return 0


def format_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:,.2f} MB"


def print_assets(title: str, filenames: list[str]) -> None:
    print(f"\n{title}")
    print("-" * 76)

    for filename in filenames:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"MISSING  | {filename}")
            continue

        kind = "DIR" if path.is_dir() else "FILE"
        size = format_size(get_size(path))

        print(f"{kind:4}     | {filename:<38} | {size}")

        if path.is_file() and path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as file:
                line_count = sum(
                    1 for line in file if line.strip()
                )

            print(f"         | Records: {line_count:,}")


def main() -> None:
    print("AI Investigation Engine — Deployment Asset Audit")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    print_assets("MAIN CORPUS AND SEARCH INDEXES", REQUIRED_ASSETS)
    print_assets("UPLOADED DOCUMENT STORAGE AND INDEXES", UPLOAD_ASSETS)
    print_assets("EVALUATION OUTPUTS", EVALUATION_ASSETS)

    if DATA_DIR.exists():
        print("\nTotal processed-data directory")
        print("-" * 76)
        print(format_size(get_size(DATA_DIR)))
    else:
        print("\nERROR: data/processed does not exist.")


if __name__ == "__main__":
    main()