
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

    direct_answer: str = Field(
        description=(
            "A concise, direct answer to the user's question. "
            "For yes/no questions, start with Yes or No. "
            "For comparisons, name the winning option and state "
            "the comparison result. For insufficient evidence, "
            "state that the answer cannot be established."
        ),
        min_length=1,
    )

    answer: str = Field(
        description=(
            "A clear explanation of the answer, grounded in "
            "the supplied evidence."
        ),
        min_length=1,
    )

    findings: list[CitedFinding] = Field(
        default_factory=list,
        description=(
            "Factual findings with exact source chunk IDs."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Missing information, unresolved contradictions, "
            "or other limits affecting the conclusion."
        ),
    )