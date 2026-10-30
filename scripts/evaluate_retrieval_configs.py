"""Compare semantic-only retrieval with semantic+keyword RRF without Gemini calls.

Requires the local data/processed/evaluation_seed.jsonl, qdrant_full collection,
and keyword_index.sqlite built by the project. Output: Evaluation_Retrieval_Comparison.md
and data/processed/retrieval_configuration_comparison.json.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient

from src.embeddings.encoder import TextEncoder
from src.evaluation.retrieval_metrics import evaluate_ranked_results, parse_supporting_facts
from src.retrieval.hybrid_search import reciprocal_rank_fusion_multiple
from src.retrieval.sqlite_bm25 import SQLiteBM25Index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEED_PATH = DATA_DIR / "evaluation_seed.jsonl"
JSON_OUTPUT = DATA_DIR / "retrieval_configuration_comparison.json"
MARKDOWN_OUTPUT = PROJECT_ROOT / "Evaluation_Retrieval_Comparison.md"
METRICS = (
    "evidence_recall@5",
    "evidence_recall@10",
    "mrr@10",
    "complete_evidence@10",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_stratified(examples: list[dict], limit: int) -> list[dict]:
    """Select question examples round-robin across available question types."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        if (
            example.get("question")
            and example.get("example_id")
            and example.get("supporting_facts")
        ):
            groups[str(example.get("question_type", "unknown"))].append(example)

    selected = []
    categories = sorted(groups)
    while len(selected) < limit:
        added = False
        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def semantic_results(client: QdrantClient, encoder: TextEncoder, question: str, limit: int) -> list[dict]:
    vector = encoder.encode_query(question)
    vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    response = client.query_points(
        collection_name="investigation_chunks",
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return [
        {
            "chunk_id": (point.payload or {}).get("chunk_id"),
            "chunk": point.payload or {},
            "score": float(point.score),
        }
        for point in response.points
        if (point.payload or {}).get("chunk_id")
    ]


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in METRICS}
    return {name: mean(row[name] for row in rows) for name in METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--candidate-limit", type=int, default=25)
    parser.add_argument("--final-limit", type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1 or args.candidate_limit < 1 or args.final_limit < 1:
        raise SystemExit("All limits must be positive integers.")

    seed = load_jsonl(SEED_PATH)
    examples = select_stratified(seed, args.limit)
    if len(examples) < 40:
        raise SystemExit(
            f"Only {len(examples)} valid questions available; at least 40 are required."
        )

    vector_path = DATA_DIR / "qdrant_full"
    keyword_path = DATA_DIR / "keyword_index.sqlite"
    if not vector_path.exists():
        raise SystemExit(f"Local Qdrant database not found: {vector_path}")
    if not keyword_path.exists():
        raise SystemExit(f"Local keyword index not found: {keyword_path}")

    client = QdrantClient(path=str(vector_path))
    keyword_index = SQLiteBM25Index(keyword_path)
    encoder = TextEncoder()
    all_rows = []

    try:
        if not client.collection_exists("investigation_chunks"):
            raise SystemExit("Qdrant collection 'investigation_chunks' is missing from the local index.")

        for position, example in enumerate(examples, start=1):
            example_id = str(example["example_id"])
            question = str(example["question"])
            gold_facts = parse_supporting_facts(example["supporting_facts"])
            if not gold_facts:
                continue

            semantic = semantic_results(
                client,
                encoder,
                question,
                args.candidate_limit,
            )
            keyword = keyword_index.search(
                question,
                top_k=args.candidate_limit,
            )
            hybrid = reciprocal_rank_fusion_multiple(
                ranked_lists=[
                    ("semantic", "semantic", semantic, 1.0),
                    ("keyword", "keyword", keyword, 1.0),
                ],
                limit=args.final_limit,
            )
            semantic_final = semantic[:args.final_limit]

            semantic_metrics = evaluate_ranked_results(
                semantic_final,
                example_id,
                gold_facts,
                recall_cutoffs=(5, 10),
                mrr_cutoff=10,
            )
            hybrid_metrics = evaluate_ranked_results(
                hybrid,
                example_id,
                gold_facts,
                recall_cutoffs=(5, 10),
                mrr_cutoff=10,
            )
            all_rows.append(
                {
                    "example_id": example_id,
                    "question_type": str(example.get("question_type", "unknown")),
                    "question": question,
                    "semantic_only": semantic_metrics,
                    "hybrid_rrf": hybrid_metrics,
                }
            )
            print(f"Evaluated {position}/{len(examples)}: {example_id}")
    finally:
        keyword_index.close()
        client.close()

    if len(all_rows) < 40:
        raise SystemExit(f"Only {len(all_rows)} examples produced valid metrics; expected at least 40.")

    semantic_rows = [row["semantic_only"] for row in all_rows]
    hybrid_rows = [row["hybrid_rrf"] for row in all_rows]
    summary = {
        "evaluation_type": "retrieval-only; no generative LLM calls",
        "question_count": len(all_rows),
        "candidate_limit": args.candidate_limit,
        "final_limit": args.final_limit,
        "semantic_only": mean_metrics(semantic_rows),
        "hybrid_rrf": mean_metrics(hybrid_rows),
        "by_question_type": {},
        "per_question": all_rows,
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[row["question_type"]].append(row)
    for question_type, rows in sorted(grouped.items()):
        summary["by_question_type"][question_type] = {
            "n": len(rows),
            "semantic_only": mean_metrics([row["semantic_only"] for row in rows]),
            "hybrid_rrf": mean_metrics([row["hybrid_rrf"] for row in rows]),
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Retrieval Configuration Comparison",
        "",
        "**Method:** Offline retrieval-only evaluation over the same questions and local corpus. No Gemini calls are made.",
        f"**Questions evaluated:** {len(all_rows)}",
        f"**Candidates per retriever:** {args.candidate_limit}",
        f"**Final result limit:** {args.final_limit}",
        "",
        "Configuration A is semantic-only Qdrant retrieval. Configuration B combines the same semantic results with SQLite BM25 keyword results using Reciprocal Rank Fusion (RRF). Metrics measure recovery of dataset-labelled supporting facts, not generated-answer correctness.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Semantic only | Hybrid RRF | Difference (Hybrid − Semantic) |",
        "|---|---:|---:|---:|",
    ]
    semantic_summary = summary["semantic_only"]
    hybrid_summary = summary["hybrid_rrf"]
    for metric in METRICS:
        a = semantic_summary[metric]
        b = hybrid_summary[metric]
        lines.append(f"| {metric} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

    lines.extend(
        [
            "",
            "## Results by question type",
            "",
            "| Type | N | Evidence recall@10: semantic | Evidence recall@10: hybrid | MRR@10: semantic | MRR@10: hybrid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for question_type, values in sorted(summary["by_question_type"].items()):
        a = values["semantic_only"]
        b = values["hybrid_rrf"]
        lines.append(
            f"| {question_type} | {values['n']} | {a['evidence_recall@10']:.4f} | "
            f"{b['evidence_recall@10']:.4f} | {a['mrr@10']:.4f} | {b['mrr@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Higher evidence recall means more labelled supporting facts appeared in the top results.",
            "- MRR@10 measures how early the first retrieved chunk containing a labelled supporting fact appears.",
            "- Complete evidence@10 is the fraction of questions for which all labelled supporting facts were recovered within the top 10.",
            "- This comparison isolates retrieval configuration. It does not measure Gemini answer-generation quality or compare multi-step planning with single-pass generation.",
            "- Re-run after any corpus, chunking, or indexing change; preserve the JSON output for per-question inspection.",
            "",
            f"Detailed per-question metrics: `data/processed/{JSON_OUTPUT.name}`",
            "",
        ]
    )
    MARKDOWN_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\nAggregate results:")
    for metric in METRICS:
        print(
            f"{metric}: semantic={semantic_summary[metric]:.4f} "
            f"hybrid={hybrid_summary[metric]:.4f}"
        )
    print(f"Markdown report: {MARKDOWN_OUTPUT}")
    print(f"Detailed JSON: {JSON_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
