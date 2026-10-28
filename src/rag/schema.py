
from typing import Literal

from pydantic import BaseModel, Field


class CitedFinding(BaseModel):
    """A factual finding supported by retrieved source chunks."""

    claim: str = Field(
        description=(
            "One factual finding or intermediate conclusion "
            "supported by the retrieved evidence."
        ),
        min_length=1,
    )

    source_ids: list[str] = Field(
        description=(
            "One or more exact chunk IDs from the supplied evidence "
            "that support this finding."
        ),
        min_length=1,
    )


class InvestigationAnswer(BaseModel):
    """Structured result of an evidence-based investigation."""

    status: Literal[
        "answered",
        "insufficient_evidence",
        "conflicting_evidence",
    ]

    answer: str = Field(
        description=(
            "A concise answer based only on the supplied evidence. "
            "Explain uncertainty when necessary."
        ),
        min_length=1,
    )

    findings: list[CitedFinding] = Field(
        default_factory=list,
        description=(
            "Supported factual findings, each accompanied by "
            "the source chunk IDs that support it."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Missing information, unresolved contradictions, "
            "or other limits affecting the conclusion."
        ),
    )
