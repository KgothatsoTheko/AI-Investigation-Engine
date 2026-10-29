
from src.evaluation.answer_metrics import (
    answer_scores,
    citation_scores,
)
from src.rag.schema import (
    CitedFinding,
    InvestigationAnswer,
)


def test_exact_match_ignores_case_and_punctuation():
    scores = answer_scores(
        "Danger: Diabolik!",
        "danger diabolik",
    )

    assert scores["answer_exact_match"] == 1.0
    assert scores["answer_token_f1"] == 1.0


def test_token_f1_rewards_partial_overlap():
    scores = answer_scores(
        "Mario Bava",
        "Mario Bava was born in 1914",
    )

    assert scores["answer_exact_match"] == 0.0
    assert 0.0 < scores["answer_token_f1"] < 1.0


def test_gold_fact_citation_metrics():
    chunk = {
        "chunk_id": "chunk-a",
        "title": "Director A",
        "text": "Director A was born in 1940.",
        "source_sentence_indices": [0],
        "provenance": [
            {
                "example_id": "example-1",
                "supporting_sentence_indices": [0],
            }
        ],
    }

    answer = InvestigationAnswer(
        status="answered",
        direct_answer="Director A",
        answer="Director A is identified by the supplied evidence.",
        findings=[
            CitedFinding(
                claim="Director A was born in 1940.",
                source_ids=["chunk-a"],
            )
        ],
    )

    scores = citation_scores(
        answer=answer,
        evidence_chunks=[chunk],
        example_id="example-1",
        supporting_facts=[["Director A", 0]],
    )

    assert scores["citation_resolution_rate"] == 1.0
    assert scores["gold_fact_citation_precision"] == 1.0
    assert scores["gold_fact_citation_recall"] == 1.0


def test_unrelated_citation_does_not_cover_gold_fact():
    chunk = {
        "chunk_id": "chunk-a",
        "title": "Unrelated Person",
        "text": "An unrelated biography.",
        "source_sentence_indices": [0],
        "provenance": [
            {
                "example_id": "example-1",
                "supporting_sentence_indices": [0],
            }
        ],
    }

    answer = InvestigationAnswer(
        status="answered",
        direct_answer="Unknown",
        answer="An unsupported claim.",
        findings=[
            CitedFinding(
                claim="The answer is supported.",
                source_ids=["chunk-a"],
            )
        ],
    )

    scores = citation_scores(
        answer=answer,
        evidence_chunks=[chunk],
        example_id="example-1",
        supporting_facts=[["Director A", 0]],
    )

    assert scores["citation_resolution_rate"] == 1.0
    assert scores["gold_fact_citation_precision"] == 0.0
    assert scores["gold_fact_citation_recall"] == 0.0
