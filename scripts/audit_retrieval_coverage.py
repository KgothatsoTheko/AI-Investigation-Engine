
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.evaluation.retrieval_metrics import (
    facts_recovered_by_chunk,
    parse_supporting_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEED_PATH = DATA_DIR / "evaluation_seed.jsonl"
RESULTS_PATH = DATA_DIR / "evaluation_answer_results.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks_sentence_grouped.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_required_chunks(
    path: Path,
    required_ids: set[str],
) -> dict[str, dict]:
    """Load only chunks referenced by the evaluation results."""
    found = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")

            if chunk_id in required_ids:
                found[chunk_id] = chunk

                if len(found) == len(required_ids):
                    break

    return found


def main() -> None:
    seed = load_jsonl(SEED_PATH)
    rows = load_jsonl(RESULTS_PATH)

    seed_by_id = {
        str(item["example_id"]): item
        for item in seed
        if item.get("example_id") is not None
    }

    successful_rows = [
        row
        for row in rows
        if row.get("evaluation_status") == "success"
    ]

    if not successful_rows:
        raise RuntimeError(
            "No successful answer evaluations found."
        )

    required_ids = {
        chunk_id
        for row in successful_rows
        for chunk_id in row.get("evidence_chunk_ids", [])
    }

    print("Loading evidence referenced by saved answers...")
    print(f"Successful answer rows: {len(successful_rows)}")
    print(f"Distinct referenced chunk IDs: {len(required_ids):,}")

    chunks_by_id = load_required_chunks(
        CHUNKS_PATH,
        required_ids,
    )

    missing_ids = required_ids - set(chunks_by_id)

    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} referenced chunk IDs "
            "were not found in the chunk file."
        )

    audited = []

    for row in successful_rows:
        example_id = str(row["example_id"])
        example = seed_by_id.get(example_id)

        if example is None:
            print(
                f"Warning: evaluation seed record not found "
                f"for {example_id}"
            )
            continue

        gold_facts = parse_supporting_facts(
            example.get("supporting_facts", [])
        )

        if not gold_facts:
            continue

        covered_facts = set()
        referenced_chunks = []

        for chunk_id in row.get("evidence_chunk_ids", []):
            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            referenced_chunks.append(chunk)

            covered_facts.update(
                facts_recovered_by_chunk(
                    chunk,
                    example_id,
                    gold_facts,
                )
            )

        recall = len(covered_facts) / len(gold_facts)

        audited.append(
            {
                "example_id": example_id,
                "question": row.get("question", ""),
                "question_type": row.get(
                    "question_type",
                    "unknown",
                ),
                "status": row.get(
                    "answer_status",
                    "unknown",
                ),
                "gold_fact_count": len(gold_facts),
                "covered_fact_count": len(covered_facts),
                "retrieved_evidence_recall": recall,
                "all_evidence_retrieved": (
                    covered_facts >= gold_facts
                ),
                "metrics": row.get("metrics", {}),
                "chunk_count": len(referenced_chunks),
            }
        )

    if not audited:
        raise RuntimeError(
            "No evaluation rows could be audited."
        )

    def print_group_summary(title: str, items: list[dict]):
        print(f"\n{title}")
        print("-" * len(title))

        if not items:
            print("No rows.")
            return

        print(f"Questions: {len(items)}")

        print(
            "Average retrieved evidence recall: "
            f"{mean(
                item['retrieved_evidence_recall']
                for item in items
            ):.3f}"
        )

        print(
            "Complete evidence available: "
            f"{mean(
                float(item['all_evidence_retrieved'])
                for item in items
            ):.1%}"
        )

        print(
            "No labelled facts recovered: "
            f"{mean(
                float(
                    item['covered_fact_count'] == 0
                )
                for item in items
            ):.1%}"
        )

    print("\nRETRIEVAL COVERAGE AUDIT")
    print("=" * 70)

    print_group_summary("All audited questions", audited)

    for status in sorted({
        item["status"]
        for item in audited
    }):
        print_group_summary(
            f"Answer status: {status}",
            [
                item for item in audited
                if item["status"] == status
            ],
        )

    print("\nCoverage by question type")
    print("-" * 70)

    for question_type in sorted({
        item["question_type"]
        for item in audited
    }):
        group = [
            item for item in audited
            if item["question_type"] == question_type
        ]

        print(
            f"{question_type}: "
            f"n={len(group)}, "
            f"mean recall="
            f"{mean(
                item['retrieved_evidence_recall']
                for item in group
            ):.3f}"
        )

    # These cases help distinguish missing retrieved evidence
    # from possible generation or abstention-policy problems.
    high_coverage_abstentions = [
        item for item in audited
        if (
            item["status"] == "insufficient_evidence"
            and item["retrieved_evidence_recall"] >= 0.8
        )
    ]

    print("\nAbstentions despite high retrieved evidence coverage")
    print("-" * 70)

    if not high_coverage_abstentions:
        print("No such cases in this evaluation sample.")

    for item in high_coverage_abstentions[:10]:
        print(
            f"\nRecall: "
            f"{item['retrieved_evidence_recall']:.2f} | "
            f"Question: {item['question']}"
        )

    # Lowest-coverage questions indicate where retrieval
    # should be investigated first.
    print("\nTen questions with the lowest evidence coverage")
    print("-" * 70)

    for item in sorted(
        audited,
        key=lambda value: value[
            "retrieved_evidence_recall"
        ],
    )[:10]:
        print(
            f"\nRecall: "
            f"{item['retrieved_evidence_recall']:.2f} | "
            f"Status: {item['status']} | "
            f"Question: {item['question']}"
        )

    output_path = DATA_DIR / "retrieval_coverage_audit.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for item in audited:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"\nDetailed audit saved to: {output_path}")


if __name__ == "__main__":
    main()
