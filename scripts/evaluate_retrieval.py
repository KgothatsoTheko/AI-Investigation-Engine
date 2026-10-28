
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.embeddings.encoder import TextEncoder
from src.evaluation.retrieval_metrics import (
    evaluate_ranked_results,
    parse_supporting_facts,
)
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.storage.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EVALUATION_PATH = DATA_DIR / "evaluation_seed.jsonl"
RESULTS_PATH = DATA_DIR / "evaluation_retrieval_results.jsonl"

QDRANT_PATH = DATA_DIR / "qdrant_full"
KEYWORD_DB_PATH = DATA_DIR / "keyword_index.sqlite"

CANDIDATE_LIMIT = 20
RECALL_CUTOFFS = (5, 10)
MRR_CUTOFF = 10


METRICS_TO_AVERAGE = (
    "evidence_recall@5",
    "evidence_recall@10",
    "mrr@10",
    "complete_evidence@10",
)


def load_jsonl(path: Path) -> list[dict]:
    """Read JSON Lines records."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def to_semantic_results(points) -> list[dict]:
    """Convert Qdrant results into the common result format."""
    results = []

    for point in points:
        payload = point.payload or {}

        results.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "chunk": payload,
                "score": float(point.score),
            }
        )

    return results


def aggregate_metrics(
    rows: list[dict],
    method: str,
) -> dict[str, float]:
    """Average the query-level metrics for one retrieval method."""
    aggregate = {}

    for metric_name in METRICS_TO_AVERAGE:
        values = [
            row["metrics"][method][metric_name]
            for row in rows
        ]

        aggregate[metric_name] = mean(values) if values else 0.0

    return aggregate


def main() -> None:
    examples = load_jsonl(EVALUATION_PATH)

    valid_examples = []

    for example in examples:
        question = example.get("question", "").strip()
        example_id = example.get("example_id")

        gold_facts = parse_supporting_facts(
            example.get("supporting_facts", [])
        )

        if not question or not example_id or not gold_facts:
            continue

        valid_examples.append(
            {
                **example,
                "question": question,
                "example_id": str(example_id),
                "_gold_facts": gold_facts,
            }
        )

    if len(valid_examples) < 40:
        raise RuntimeError(
            "At least 40 valid evaluation questions are required. "
            f"Found {len(valid_examples)}."
        )

    print("Retrieval evaluation")
    print("=" * 60)
    print(f"Evaluation seed records: {len(examples)}")
    print(f"Questions with supporting facts: {len(valid_examples)}")

    question_types = Counter(
        item.get("question_type", "unknown")
        for item in valid_examples
    )

    print("\nQuestion types:")
    for question_type, count in sorted(question_types.items()):
        print(f"  {question_type}: {count}")

    print("\nLoading embedding model and indexes...")
    encoder = TextEncoder()

    keyword_index = SQLiteBM25Index(
        KEYWORD_DB_PATH
    )
    vector_store = QdrantVectorStore(
        database_path=QDRANT_PATH,
        collection_name="investigation_chunks",
    )

    rows = []

    try:
        keyword_count = keyword_index.count_documents()
        vector_count = vector_store.count_points()

        print(f"Keyword-index chunks: {keyword_count:,}")
        print(f"Vector-index chunks: {vector_count:,}")

        if keyword_count != vector_count:
            raise RuntimeError(
                "Keyword and vector indexes have different counts. "
                "Rebuild or inspect the indexes before evaluation."
            )

        questions = [
            item["question"]
            for item in valid_examples
        ]

        query_vectors = encoder.encode_queries(
            questions,
            batch_size=16,
        )

        for position, (example, query_vector) in enumerate(
            zip(valid_examples, query_vectors),
            start=1,
        ):
            question = example["question"]
            example_id = example["example_id"]
            gold_facts = example["_gold_facts"]

            semantic_results = to_semantic_results(
                vector_store.search(
                    query_vector,
                    limit=CANDIDATE_LIMIT,
                )
            )

            keyword_results = keyword_index.search(
                question,
                top_k=CANDIDATE_LIMIT,
            )

            hybrid_results = reciprocal_rank_fusion(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                limit=CANDIDATE_LIMIT,
            )

            result_sets = {
                "semantic": semantic_results,
                "keyword": keyword_results,
                "hybrid": hybrid_results,
            }

            metrics = {
                method: evaluate_ranked_results(
                    results=results,
                    example_id=example_id,
                    gold_facts=gold_facts,
                    recall_cutoffs=RECALL_CUTOFFS,
                    mrr_cutoff=MRR_CUTOFF,
                )
                for method, results in result_sets.items()
            }

            rows.append(
                {
                    "example_id": example_id,
                    "question": question,
                    "question_type": example.get(
                        "question_type",
                        "unknown",
                    ),
                    "gold_fact_count": len(gold_facts),
                    "metrics": metrics,
                    "top_hybrid_results": [
                        {
                            "title": result["chunk"].get(
                                "title",
                                "Unknown",
                            ),
                            "chunk_id": result.get("chunk_id"),
                            "hybrid_score": result[
                                "hybrid_score"
                            ],
                        }
                        for result in hybrid_results[:5]
                    ],
                }
            )

            if position % 10 == 0:
                print(
                    f"Evaluated {position}/"
                    f"{len(valid_examples)} questions."
                )

    finally:
        keyword_index.close()
        vector_store.close()

    write_jsonl(RESULTS_PATH, rows)

    print("\nAggregate retrieval metrics")
    print("=" * 60)

    summaries = {}

    for method in ("semantic", "keyword", "hybrid"):
        summary = aggregate_metrics(rows, method)
        summaries[method] = summary

        print(f"\n{method.upper()}")

        for metric_name, value in summary.items():
            print(f"  {metric_name}: {value:.4f}")

    print("\nFive lowest hybrid Evidence Recall@10 questions")
    print("=" * 60)

    lowest = sorted(
        rows,
        key=lambda row: row["metrics"]["hybrid"][
            "evidence_recall@10"
        ],
    )[:5]

    for item in lowest:
        recall = item["metrics"]["hybrid"][
            "evidence_recall@10"
        ]

        print(f"\nRecall@10: {recall:.2f}")
        print(f"Question: {item['question']}")
        print(f"Supporting facts: {item['gold_fact_count']}")

    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
