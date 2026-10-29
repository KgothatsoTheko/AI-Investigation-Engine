
import pytest

from src.rag.generator import (
    InvalidCitationError,
    validate_citations,
)
from src.rag.schema import (
    CitedFinding,
    InvestigationAnswer,
)


def sample_evidence():
    return [
        {
            "chunk_id": "chunk-a",
            "title": "Director A",
            "text": "Director A was born in 1940.",
        },
        {
            "chunk_id": "chunk-b",
            "title": "Director B",
            "text": "Director B was born in 1960.",
        },
    ]


def test_known_citations_are_accepted():
    answer = InvestigationAnswer(
        status="answered",
        direct_answer="Director B",
        answer="Director B was younger.",
        findings=[
            CitedFinding(
                claim="Director A was born in 1940.",
                source_ids=["chunk-a"],
            ),
            CitedFinding(
                claim="Director B was born in 1960.",
                source_ids=["chunk-b"],
            ),
        ],
    )

    validated = validate_citations(
        answer,
        sample_evidence(),
    )

    assert validated.status == "answered"
    assert len(validated.findings) == 2


def test_unknown_citation_is_rejected():
    answer = InvestigationAnswer(
        status="answered",
        direct_answer="The answer is supported by evidence.",
        answer="The answer is supported by evidence.",
        findings=[
            CitedFinding(
                claim="A factual finding.",
                source_ids=["chunk-does-not-exist"],
            ),
        ],
    )

    with pytest.raises(
        InvalidCitationError,
        match="unknown source IDs",
    ):
        validate_citations(
            answer,
            sample_evidence(),
        )


def test_answered_status_requires_cited_findings():
    answer = InvestigationAnswer(
        status="answered",
        direct_answer="An answer without supporting findings.",
        answer="An answer without supporting findings.",
        findings=[],
    )

    with pytest.raises(
        InvalidCitationError,
        match="without any cited findings",
    ):
        validate_citations(
            answer,
            sample_evidence(),
        )


def test_insufficient_evidence_can_have_no_findings():
    answer = InvestigationAnswer(
        status="insufficient_evidence",
        direct_answer="There is not enough information.",
        answer="There is not enough information.",
        findings=[],
        limitations=["The required fact was not found."],
    )

    validated = validate_citations(
        answer,
        sample_evidence(),
    )

    assert validated.status == "insufficient_evidence"
    assert validated.findings == []
