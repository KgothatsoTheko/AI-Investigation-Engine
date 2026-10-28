
from pathlib import Path

from src.rag.generator import InvalidCitationError
from src.rag.investigator import MultiStepInvestigator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def display_result(result: dict) -> None:
    answer = result["answer"]
    evidence_chunks = result["evidence_chunks"]

    evidence_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in evidence_chunks
    }

    print("\n" + "=" * 70)
    print("INVESTIGATION TRACE")
    print("=" * 70)

    for number, item in enumerate(
        result["search_log"],
        start=1,
    ):
        print(f"\nSearch {number}: {item['query']}")
        print(f"Results returned: {item['results_returned']}")
        print(f"New unique chunks: {item['new_unique_chunks']}")

    print(f"\nTotal evidence chunks: {len(evidence_chunks)}")

    print("\n" + "=" * 70)
    print("INVESTIGATION RESULT")
    print("=" * 70)

    print(f"\nStatus: {answer.status}")
    print(f"\nAnswer:\n{answer.answer}")

    print("\nSupporting findings")
    print("-" * 70)

    for number, finding in enumerate(
        answer.findings,
        start=1,
    ):
        print(f"\n{number}. {finding.claim}")

        for source_id in finding.source_ids:
            source = evidence_by_id[source_id]

            print(
                f"  [{source_id}] "
                f"{source.get('title', 'Unknown title')}"
            )

            print(
                "    " + source.get("text", "")[:400]
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

    investigator = MultiStepInvestigator(
        data_dir=DATA_DIR,
        max_search_calls=4,
    )

    try:
        print("\nStarting multi-step investigation...")

        result = investigator.investigate(question)
        display_result(result)

    except InvalidCitationError as exc:
        print("\nCitation validation failed.")
        print(f"Reason: {exc}")
        print("The answer was not displayed as trustworthy.")

    finally:
        investigator.close()


if __name__ == "__main__":
    main()
