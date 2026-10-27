
import json

from src.ingestion.corpus_store import SQLiteCorpusStore


def make_document(provenance):
    return {
        "document_id": "doc-123",
        "title": "Example Title",
        "text": "This is an example passage.",
        "metadata": {
            "dataset": "2WikiMultihopQA",
            "split": "train",
        },
        "provenance": provenance,
    }


def test_duplicate_passages_merge_provenance(tmp_path):
    store = SQLiteCorpusStore(
        tmp_path / "test_corpus.sqlite"
    )

    first_source = {
        "example_id": "example-a",
        "passage_index": 0,
        "supporting_sentence_indices": [0],
    }

    second_source = {
        "example_id": "example-b",
        "passage_index": 3,
        "supporting_sentence_indices": [],
    }

    try:
        # First occurrence: insert a new passage.
        assert store.add_document(
            make_document([first_source])
        ) is True

        # Second occurrence: do not insert another passage.
        assert store.add_document(
            make_document([second_source])
        ) is False

        # Repeating the same provenance should not duplicate it.
        assert store.add_document(
            make_document([first_source, second_source])
        ) is False

        assert store.count_documents() == 1

        output_path = tmp_path / "corpus.jsonl"
        store.export_jsonl(output_path)

        lines = output_path.read_text(
            encoding="utf-8"
        ).splitlines()

        assert len(lines) == 1

        stored = json.loads(lines[0])

        assert stored["document_id"] == "doc-123"
        assert stored["text"] == "This is an example passage."
        assert len(stored["provenance"]) == 2

        stored_sources = {
            item["example_id"]
            for item in stored["provenance"]
        }

        assert stored_sources == {"example-a", "example-b"}

    finally:
        store.close()


def test_invalid_document_is_rejected(tmp_path):
    store = SQLiteCorpusStore(
        tmp_path / "test_invalid.sqlite"
    )

    try:
        try:
            store.add_document({"title": "Missing ID"})
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError for an invalid document."
            )

        assert store.count_documents() == 0

    finally:
        store.close()
