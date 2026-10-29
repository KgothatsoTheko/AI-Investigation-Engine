import inspect
from pathlib import Path

from src.ingestion.chunking import chunk_document
from src.embeddings.encoder import TextEncoder
from src.storage.vector_store import QdrantVectorStore
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.rag.retriever import HybridEvidenceRetriever


def print_source(label, obj):
    print(f"\n{'=' * 76}")
    print(label)
    print("=" * 76)

    try:
        print(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        print(f"Source unavailable: {exc}")


def inspect_class(cls, method_names):
    for method_name in method_names:
        method = getattr(cls, method_name, None)

        if method is not None:
            print_source(f"{cls.__name__}.{method_name}", method)


if __name__ == "__main__":
    print_source("chunk_document", chunk_document)

    inspect_class(
        TextEncoder,
        [
            "__init__",
            "encode_documents",
            "encode_queries",
            "encode_query",
            "get_embedding_dimension",
        ],
    )

    inspect_class(
        QdrantVectorStore,
        [
            "__init__",
            "_point_id",
            "recreate_collection",
            "upsert_batch",
            "search",
            "count_points",
            "close",
        ],
    )

    inspect_class(
        SQLiteBM25Index,
        [
            "__init__",
            "add_documents",
            "search",
            "count_documents",
            "close",
        ],
    )

    print_source("reciprocal_rank_fusion", reciprocal_rank_fusion)

    inspect_class(
        HybridEvidenceRetriever,
        [
            "__init__",
            "retrieve",
            "close",
        ],
    )

    for filename in [
        "scripts/index_corpus.py",
        "scripts/build_keyword_index.py",
    ]:
        path = Path(filename)

        print(f"\n{'=' * 76}")
        print(f"FILE: {filename}")
        print("=" * 76)

        if path.exists():
            print(path.read_text(encoding="utf-8"))
        else:
            print("File not found.")