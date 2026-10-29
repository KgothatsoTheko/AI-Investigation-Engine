
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_answer_results.jsonl"
)


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    rows = load_results()

    successful = [
        row for row in rows
        if row.get("evaluation_status") == "success"
    ]

    print("ANSWER EVALUATION AUDIT")
    print("=" * 70)
    print(f"Total saved rows: {len(rows)}")
    print(f"Successful rows: {len(successful)}")

    if not successful:
        print("No successful evaluations to inspect.")
        return

    statuses = Counter(
        row.get("answer_status", "unknown")
        for row in successful
    )

    print("\nAnswer status distribution")
    for status, count in sorted(statuses.items()):
        print(f"{status}: {count} ({count / len(successful):.1%})")

    by_type = defaultdict(list)

    for row in successful:
        by_type[row.get("question_type", "unknown")].append(row)

    print("\nMetrics by question type")

    for question_type, group in sorted(by_type.items()):
        exact_match = mean(
            item["metrics"]["answer_exact_match"]
            for item in group
        )

        token_f1 = mean(
            item["metrics"]["answer_token_f1"]
            for item in group
        )

        answered_rate = mean(
            item["metrics"]["answered_rate_on_supported_questions"]
            for item in group
        )

        print(
            f"{question_type}: n={len(group)}, "
            f"exact_match={exact_match:.3f}, "
            f"token_f1={token_f1:.3f}, "
            f"answered_rate={answered_rate:.3f}"
        )

    # Print the weakest answers first so the failures are easier
    # to inspect without dumping every full source passage.
    weakest = sorted(
        successful,
        key=lambda row: (
            row["metrics"]["answer_token_f1"],
            row["metrics"]["answer_exact_match"],
        ),
    )[:15]

    print("\n15 answers with the lowest token F1")
    print("=" * 70)

    for number, row in enumerate(weakest, start=1):
        metrics = row["metrics"]

        print(f"\n{number}. {row['question']}")
        print(f"Type: {row.get('question_type')}")
        print(f"Status: {row.get('answer_status')}")

        print(f"Reference: {row.get('reference_answer')}")
        print(f"Predicted: {row.get('predicted_direct_answer')}")

        print(
            f"Exact match: {metrics['answer_exact_match']:.2f} | "
            f"Token F1: {metrics['answer_token_f1']:.2f}"
        )

        print(
            "Gold-fact citation precision: "
            f"{metrics['gold_fact_citation_precision']:.2f}"
        )

        if row.get("limitations"):
            print("Limitations:", "; ".join(row["limitations"]))


if __name__ == "__main__":
    main()
