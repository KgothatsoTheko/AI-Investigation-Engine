
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


from src.evaluation.answer_metrics import (
    answer_scores,
    citation_scores,
)
from src.rag.generator import GeminiAnswerGenerator
from src.rag.retriever import HybridEvidenceRetriever

import os

from google.genai.errors import ClientError, ServerError
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "evaluation_seed.jsonl"
OUTPUT_PATH = DATA_DIR / "evaluation_answer_results.jsonl"


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


def select_stratified(
    examples: list[dict],
    limit: int,
) -> list[dict]:
    """
    Select questions in round-robin order across question types.

    This avoids selecting only one type when the seed is ordered
    by type. Available categories may still be imbalanced.
    """
    groups = defaultdict(list)

    for example in examples:
        if (
            example.get("question")
            and example.get("example_id")
            and example.get("answer") is not None
            and example.get("supporting_facts")
        ):
            groups[
                example.get("question_type", "unknown")
            ].append(example)

    selected = []

    selected_types = Counter(
    example.get("question_type", "unknown")
    for example in selected
    )

    print("\nSelected question types:")

    for question_type, count in sorted(selected_types.items()):
        print(f"  {question_type}: {count}")

    categories = sorted(groups)

    while len(selected) < limit:
        added_this_round = False

        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop(0))
                added_this_round = True

                if len(selected) >= limit:
                    break

        if not added_this_round:
            break

    return selected


def aggregate(
    successful_rows: list[dict],
) -> dict[str, float]:
    metric_names = [
        "answer_exact_match",
        "answer_token_f1",
        "citation_resolution_rate",
        "gold_fact_citation_precision",
        "gold_fact_citation_recall",
        "answered_rate_on_supported_questions",
    ]

    return {
        name: mean(
            row["metrics"][name]
            for row in successful_rows
        )
        for name in metric_names
    }



def generate_answer_with_fallback(
    generator,
    question,
    evidence_chunks,
):
    """Use a fallback model after a server error or HTTP 429."""
    try:
        return generator.answer(question, evidence_chunks)

    except (ServerError, ClientError) as exc:
        # Retry with another model only for server errors or 429s.
        if isinstance(exc, ClientError) and exc.code != 429:
            raise

        fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL",
            "gemini-3.5-flash-lite",
        ).strip()

        primary_model = getattr(generator, "model_name", "")

        if not fallback_model or fallback_model == primary_model:
            raise

        print(
            f"\nPrimary model failed with HTTP {exc.code}. "
            f"Trying fallback model: {fallback_model}"
        )

        fallback_generator = GeminiAnswerGenerator(
            model_name=fallback_model,
        )

        try:
            return fallback_generator.answer(
                question,
                evidence_chunks,
            )
        finally:
            fallback_generator.client.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of evaluation questions (default: 40).",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start the answer evaluation from scratch.",
    )

    args = parser.parse_args()

    if args.limit < 40:
        print(
            "Note: the assignment requires at least 40 test "
            "questions in the final evaluation report."
        )

    if args.reset and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    seed = load_jsonl(INPUT_PATH)

    selected = select_stratified(
        seed,
        limit=args.limit,
    )

    if len(selected) < 40:
        raise RuntimeError(
            f"Only {len(selected)} valid questions were selected. "
            "At least 40 are required."
        )

    selected_ids = {
        str(example["example_id"])
        for example in selected
    }

    existing = (
        load_jsonl(OUTPUT_PATH)
        if OUTPUT_PATH.exists()
        else []
    )

    completed_ids = {
        str(row["example_id"])
        for row in existing
        if row.get("evaluation_status") == "success"
    }

    print("Answer-quality evaluation")
    print("=" * 60)
    print(f"Seed size: {len(seed)}")
    print(f"Selected questions: {len(selected)}")
    print(f"Already completed: "
          f"{len(completed_ids & selected_ids)}")
    print(f"Remaining: "
          f"{len(selected_ids - completed_ids)}")
    print(
        "This run makes one answer-generation API request "
        "per remaining question."
    )

    retriever = HybridEvidenceRetriever(
        data_dir=DATA_DIR,
        candidate_limit=25,
        final_limit=12,
    )

    generator = GeminiAnswerGenerator()

    try:
        for position, example in enumerate(selected, start=1):
            example_id = str(example["example_id"])

            if example_id in completed_ids:
                continue

            question = example["question"]
            expected_answer = str(example["answer"])

            print(
                f"\nEvaluating {position}/{len(selected)}: "
                f"{question}"
            )

            evidence_results = retriever.retrieve(question)

            evidence_chunks = [
                item["chunk"]
                for item in evidence_results
            ]

            answer = generate_answer_with_fallback(
                generator,
                question,
                evidence_chunks,
            )

            scores = answer_scores(
                answer.direct_answer,
                expected_answer,
            )

            scores.update(
                citation_scores(
                    answer=answer,
                    evidence_chunks=evidence_chunks,
                    example_id=example_id,
                    supporting_facts=example[
                        "supporting_facts"
                    ],
                )
            )

            # Every question in this benchmark has a reference
            # answer and supporting facts. An abstention therefore
            # counts as a missed answerable question.
            scores[
                "answered_rate_on_supported_questions"
            ] = float(answer.status == "answered")

            row = {
                "evaluation_status": "success",
                "example_id": example_id,
                "question": question,
                "question_type": example.get("question_type", "unknown"),
                "reference_answer": expected_answer,
                "predicted_direct_answer": answer.direct_answer,
                "answer_status": answer.status,
                "findings": [
                    finding.model_dump()
                    for finding in answer.findings
                ],
                "limitations": answer.limitations,
                "evidence_chunk_ids": [
                    chunk["chunk_id"]
                    for chunk in evidence_chunks
                ],
                "metrics": scores,
            }

            # Save after each question so a later failure does not
            # discard results that have already completed.
            append_jsonl(OUTPUT_PATH, row)
            completed_ids.add(example_id)

            print(
                f"Direct answer: {answer.direct_answer}"
            )
            print(
                "Exact match: "
                f"{scores['answer_exact_match']:.2f} | "
                "Token F1: "
                f"{scores['answer_token_f1']:.2f} | "
                "Gold citation precision: "
                f"{scores['gold_fact_citation_precision']:.2f}"
            )

    finally:
        retriever.close()
        generator.client.close()

    # Aggregate only successful rows belonging to this run's
    # currently selected question set.
    all_rows = (
        load_jsonl(OUTPUT_PATH)
        if OUTPUT_PATH.exists()
        else []
    )

    successful_rows = [
        row
        for row in all_rows
        if (
            row.get("evaluation_status") == "success"
            and str(row.get("example_id")) in selected_ids
        )
    ]

    if not successful_rows:
        raise RuntimeError("No successful answer evaluations.")

    summary = aggregate(successful_rows)

    print("\nAggregate answer-quality metrics")
    print("=" * 60)

    for name, value in summary.items():
        print(f"{name}: {value:.4f}")

    print(
        f"\nSuccessful questions: "
        f"{len(successful_rows)}/{len(selected)}"
    )
    print(f"Detailed results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
