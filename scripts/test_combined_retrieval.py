from pathlib import Path

from src.rag.retriever import HybridEvidenceRetriever


def main():
    retriever = HybridEvidenceRetriever(
        data_dir=Path("data/processed"),
    )

    try:
        question = (
            "What are the requirements for the AI Investigation Engine?"
        )

        results = retriever.retrieve(question)

        print(f"Question: {question}")
        print(f"Evidence chunks returned: {len(results)}\n")

        for index, result in enumerate(results, start=1):
            chunk = result["chunk"]

            print(f"[{index}] {chunk.get('title', 'Untitled')}")
            print(
                f"Document ID: {chunk.get('document_id', 'unknown')}"
            )
            print(
                f"Chunk ID: {chunk.get('chunk_id', 'unknown')}"
            )
            print(
                f"Fusion score: {result['hybrid_score']:.5f}"
            )
            print(chunk.get("text", "")[:700])
            print()

    finally:
        retriever.close()


if __name__ == "__main__":
    main()