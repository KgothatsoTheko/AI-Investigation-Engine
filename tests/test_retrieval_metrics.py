
from src.evaluation.retrieval_metrics import (
    evaluate_ranked_results,
    parse_supporting_facts,
)


def make_result(title, chunk_id, example_id, indices):
    return {
        "chunk_id": chunk_id,
        "chunk": {
            "chunk_id": chunk_id,
            "title": title,
            "source_sentence_indices": indices,
            "provenance": [
                {
                    "example_id": example_id,
                    "supporting_sentence_indices": indices,
                }
            ],
        },
    }


def test_parse_supporting_facts_normalizes_titles():
    facts = parse_supporting_facts(
        [
            ["Mario Bava", 0],
            ["MARIO-BAVA", 1],
        ]
    )

    assert facts == {
        ("mariobava", 0),
        ("mariobava", 1),
    }


def test_evidence_recall_and_mrr():
    gold_facts = parse_supporting_facts(
    [
        ["Director A", 0],
        ["Director B", 1],
    ]
)

    results = [
        # First result is not supporting evidence.
        make_result(
            "Unrelated Passage",
            "other",
            "example-1",
            [0],
        ),
        make_result(
            "Director A",
            "a",
            "example-1",
            [0],
        ),
        make_result(
            "Director B",
            "b",
            "example-1",
            [1],
        ),
    ]

    metrics = evaluate_ranked_results(
        results=results,
        example_id="example-1",
        gold_facts=gold_facts,
    )

    assert metrics["evidence_recall@5"] == 1.0
    assert metrics["evidence_recall@10"] == 1.0
    assert metrics["mrr@10"] == 0.5
    assert metrics["complete_evidence@10"] == 1.0


def test_provenance_must_match_evaluation_example():
    gold_facts = {("director a", 0)}

    results = [
        make_result(
            "Director A",
            "a",
            "different-example",
            [0],
        )
    ]

    metrics = evaluate_ranked_results(
        results=results,
        example_id="example-1",
        gold_facts=gold_facts,
    )

    assert metrics["evidence_recall@5"] == 0.0
    assert metrics["mrr@10"] == 0.0
    assert metrics["complete_evidence@10"] == 0.0
