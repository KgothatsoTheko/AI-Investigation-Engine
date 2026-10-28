
from pathlib import Path

from src.rag.generator import (
    GeminiAnswerGenerator,
    InvalidCitationError,
)
from src.rag.retriever import HybridEvidenceRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def display_answer(answer, evidence_chunks: list[dict]) -> None:
    """Print the answer and its verified source references."""
    evidence_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in evidence_chunks
    }

    print("\n" + "=" * 70)
    print("INVESTIGATION RESULT")
    print("=" * 70)

    print(f"\nStatus: {answer.status}")
    print(f"\nAnswer:\n{answer.answer}")

    if answer.findings:
        print("\nSupporting findings")
        print("-" * 70)

        for number, finding in enumerate(
            answer.findings,
            start=1,
        ):
            print(f"\n{number}. {finding.claim}")
            print("Sources:")

            for source_id in finding.source_ids:
                chunk = evidence_by_id[source_id]

                print(
                    f"  [{source_id}] "
                    f"{chunk.get('title', 'Unknown title')}"
                )

                print(
                    "    Text: "
                    + chunk.get("text", "")[:400]
                )

    if answer.limitations:
        print("\nLimitations and uncertainties")
        for limitation in answer.limitations:
            print(f"- {limitation}")


def main() -> None:
    question = input(
        "What would you like to investigate? "
    ).strip()

    if not question:
        print("Please enter a non-empty question.")
        return

    retriever = HybridEvidenceRetriever(
        data_dir=DATA_DIR,
        candidate_limit=25,
        final_limit=12,
    )

    try:
        print("\nSearching the full corpus...")

        retrieved_results = retriever.retrieve(question)

        evidence_chunks = [
            result["chunk"]
            for result in retrieved_results
        ]

        print(
            f"Selected {len(evidence_chunks)} evidence chunks "
            "for the investigation."
        )

        generator = GeminiAnswerGenerator()

        try:
            answer = generator.answer(
                question,
                evidence_chunks,
            )

        except InvalidCitationError as exc:
            print("\nThe generated answer failed citation validation.")
            print(f"Reason: {exc}")
            print(
                "The answer was not displayed because its "
                "citations could not be verified."
            )
            return

        display_answer(answer, evidence_chunks)

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
