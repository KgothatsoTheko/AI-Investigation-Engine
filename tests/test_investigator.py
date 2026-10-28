
from src.rag.investigator import (
    content_identity,
    select_diverse_evidence,
)


def make_chunk(chunk_id, title, text):
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
    }


def test_content_identity_ignores_punctuation_and_case():
    first = make_chunk(
        "chunk-a",
        "Mario Bava",
        "Italian FILM director.",
    )

    second = make_chunk(
        "chunk-b",
        "MARIO-BAVA",
        "Italian film director!",
    )

    assert content_identity(first) == content_identity(second)


def test_evidence_selection_preserves_search_diversity():
    key_a1 = ("entity a", "first evidence")
    key_a2 = ("entity a", "second evidence")
    key_b1 = ("entity b", "first evidence")
    key_b2 = ("entity b", "second evidence")
    key_c1 = ("entity c", "third evidence")

    evidence = {
        key_a1: make_chunk("a1", "Entity A", "First evidence"),
        key_a2: make_chunk("a2", "Entity A", "Second evidence"),
        key_b1: make_chunk("b1", "Entity B", "First evidence"),
        key_b2: make_chunk("b2", "Entity B", "Second evidence"),
        key_c1: make_chunk("c1", "Entity C", "Third evidence"),
    }

    search_rounds = [
        [key_a1, key_a2],
        [key_b1, key_b2],
        [key_a1, key_c1],
    ]

    selected = select_diverse_evidence(
        search_rounds,
        evidence,
        limit=5,
    )

    assert [item["chunk_id"] for item in selected] == [
        "a1",
        "b1",
        "a2",
        "b2",
        "c1",
    ]

    # Duplicate evidence isn't included twice.
    assert len({
        content_identity(item)
        for item in selected
    }) == 5


def test_evidence_selection_respects_limit():
    key_a = ("title a", "text a")
    key_b = ("title b", "text b")

    evidence = {
        key_a: make_chunk("a", "Title A", "Text A"),
        key_b: make_chunk("b", "Title B", "Text B"),
    }

    selected = select_diverse_evidence(
        [[key_a, key_b]],
        evidence,
        limit=1,
    )

    assert len(selected) == 1
    assert selected[0]["chunk_id"] == "a"


def test_evidence_selection_rejects_invalid_limit():
    try:
        select_diverse_evidence([], {}, limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for limit=0."
        )
