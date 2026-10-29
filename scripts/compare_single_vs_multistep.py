
import argparse
import json
from pathlib import Path

from src.evaluation.retrieval_metrics import (
    facts_recovered_by_chunk,
    parse_supporting_facts,
)
from src.rag.investigator import MultiStepInvestigator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEED_PATH = DATA_DIR / "evaluation_seed.jsonl"
ANSWER_RESULTS_PATH = DATA_DIR / "evaluation_answer_results.jsonl"
COVERAGE_AUDIT_PATH = DATA_DIR / "retrieval_coverage_audit.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"

OUTPUT_PATH = DATA_DIR / "single_vs_multistep_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def measure_coverage(
    chunks: list[dict],
    example_id: str,
    gold_facts: set[tuple[str, int]],
) -> dict:
    covered = set()

    for chunk in chunks:
        covered.update(
            facts_recovered_by_chunk(
                chunk,
                example_id,
                gold_facts,
            )
        )

    return {
        "supporting_facts_found": len(covered),
        "supporting_facts_total": len(gold_facts),
        "evidence_recall": (
            len(covered) / len(gold_facts)
            if gold_facts else 0.0
        ),
        "complete_evidence": covered >= gold_facts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of difficult questions to compare.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard this comparison's previous results.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    if args.reset and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    seed = load_jsonl(SEED_PATH)
    answer_rows = load_jsonl(ANSWER_RESULTS_PATH)
    audit_rows = load_jsonl(COVERAGE_AUDIT_PATH)

    seed_by_id = {
        str(item["example_id"]): item
        for item in seed
        if item.get("example_id") is not None
    }

    answer_by_id = {
        str(item["example_id"]): item
        for item in answer_rows
        if (
            item.get("evaluation_status") == "success"
            and item.get("example_id") is not None
        )
    }

    # Select the lowest-coverage cases from the existing audit.
    difficult_cases = sorted(
        (
            item for item in audit_rows
            if str(item["example_id"]) in answer_by_id
            and str(item["example_id"]) in seed_by_id
        ),
        key=lambda item: item["retrieved_evidence_recall"],
    )[:args.limit]

    if not difficult_cases:
        raise RuntimeError(
            "No matching cases were found in the retrieval audit."
        )

    existing = (
        load_jsonl(OUTPUT_PATH)
        if OUTPUT_PATH.exists()
        else []
    )

    completed_ids = {
        str(row["example_id"])
        for row in existing
        if row.get("status") == "success"
    }

    # Load the chunk records once so the baseline evidence IDs
    # can be resolved back to their actual source metadata.
    print("Loading source chunks...")
    chunk_records = load_jsonl(CHUNKS_PATH)

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunk_records
        if chunk.get("chunk_id")
    }

    print(f"Chunks available: {len(chunks_by_id):,}")
    print(f"Cases selected: {len(difficult_cases)}")
    print(f"Already completed: {len(completed_ids)}")

    investigator = MultiStepInvestigator(
        data_dir=DATA_DIR,
        max_search_calls=4,
    )

    try:
        for position, audit_item in enumerate(
            difficult_cases,
            start=1,
        ):
            example_id = str(audit_item["example_id"])

            if example_id in completed_ids:
                print(f"Skipping completed case {example_id}.")
                continue

            example = seed_by_id[example_id]
            baseline = answer_by_id[example_id]

            question = example["question"]
            gold_facts = parse_supporting_facts(
                example.get("supporting_facts", [])
            )

            if not gold_facts:
                print(
                    f"Skipping {example_id}: no supporting facts."
                )
                continue

            baseline_chunks = [
                chunks_by_id[chunk_id]
                for chunk_id in baseline.get(
                    "evidence_chunk_ids",
                    [],
                )
                if chunk_id in chunks_by_id
            ]

            single_pass_metrics = measure_coverage(
                baseline_chunks,
                example_id,
                gold_facts,
            )

            print(
                f"\nCase {position}/{len(difficult_cases)}"
            )
            print(f"Question: {question}")
            print(
                "Single-pass evidence recall: "
                f"{single_pass_metrics['evidence_recall']:.2f}"
            )

            try:
                result = investigator.investigate(question)

                multi_step_chunks = result["evidence_chunks"]

                multi_step_metrics = measure_coverage(
                    multi_step_chunks,
                    example_id,
                    gold_facts,
                )

                answer = result["answer"]

                row = {
                    "status": "success",
                    "example_id": example_id,
                    "question": question,
                    "question_type": example.get("type"),
                    "single_pass": single_pass_metrics,
                    "multi_step": multi_step_metrics,
                    "recall_change": (
                        multi_step_metrics["evidence_recall"]
                        - single_pass_metrics["evidence_recall"]
                    ),
                    "multi_step_answer_status": answer.status,
                    "multi_step_direct_answer": answer.direct_answer,
                    "search_log": result["search_log"],
                    "multi_step_chunk_ids": [
                        chunk["chunk_id"]
                        for chunk in multi_step_chunks
                    ],
                }

                append_jsonl(OUTPUT_PATH, row)

                print(
                    "Multi-step evidence recall: "
                    f"{multi_step_metrics['evidence_recall']:.2f}"
                )
                print(
                    f"Change: {row['recall_change']:+.2f}"
                )
                print(
                    f"Final answer status: {answer.status}"
                )

            except Exception as exc:
                # Save the error so it can be inspected, but don't
                # treat a failed API call as a retrieval failure.
                append_jsonl(
                    OUTPUT_PATH,
                    {
                        "status": "error",
                        "example_id": example_id,
                        "question": question,
                        "error": str(exc),
                        "single_pass": single_pass_metrics,
                    },
                )

                print(f"Case failed: {exc}")
                print(
                    "This case is excluded from the successful "
                    "multi-step retrieval comparison."
                )

    finally:
        investigator.close()

    # Report the successful comparisons.
    comparison_rows = [
        row for row in load_jsonl(OUTPUT_PATH)
        if row.get("status") == "success"
        and str(row.get("example_id")) in {
            str(item["example_id"])
            for item in difficult_cases
        }
    ]

    print("\nCOMPARISON SUMMARY")
    print("=" * 60)
    print(f"Successful comparisons: {len(comparison_rows)}")

    if comparison_rows:
        from statistics import mean

        print(
            "Mean single-pass evidence recall: "
            f"{mean(
                row['single_pass']['evidence_recall']
                for row in comparison_rows
            ):.3f}"
        )

        print(
            "Mean multi-step evidence recall: "
            f"{mean(
                row['multi_step']['evidence_recall']
                for row in comparison_rows
            ):.3f}"
        )

        print(
            "Mean recall change: "
            f"{mean(
                row['recall_change']
                for row in comparison_rows
            ):+.3f}"
        )

    print(f"Detailed results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
