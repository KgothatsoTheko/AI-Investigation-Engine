
import re
from collections import Counter
from typing import Any

from src.evaluation.retrieval_metrics import (
    facts_recovered_by_chunk,
    parse_supporting_facts,
)
from src.rag.schema import InvestigationAnswer


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact-match evaluation."""
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


def answer_scores(
    predicted: str,
    expected: str,
) -> dict[str, float]:
    """
    Calculate exact match and token-level F1.

    Token F1 measures overlap between the normalized predicted
    answer and the reference answer. It does not judge whether
    a long narrative explanation is logically correct.
    """
    predicted_normalized = normalize_answer(predicted)
    expected_normalized = normalize_answer(expected)

    exact_match = float(
        predicted_normalized == expected_normalized
    )

    predicted_tokens = predicted_normalized.split()
    expected_tokens = expected_normalized.split()

    if not predicted_tokens and not expected_tokens:
        token_f1 = 1.0

    elif not predicted_tokens or not expected_tokens:
        token_f1 = 0.0

    else:
        predicted_counts = Counter(predicted_tokens)
        expected_counts = Counter(expected_tokens)

        common = sum(
            (predicted_counts & expected_counts).values()
        )

        if common == 0:
            token_f1 = 0.0

        else:
            precision = common / len(predicted_tokens)
            recall = common / len(expected_tokens)

            token_f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "answer_exact_match": exact_match,
        "answer_token_f1": token_f1,
    }


def citation_scores(
    answer: InvestigationAnswer,
    evidence_chunks: list[dict[str, Any]],
    example_id: str,
    supporting_facts: Any,
) -> dict[str, float]:
    """
    Evaluate cited sources against the dataset's gold facts.

    This is a reference-based citation metric, not a complete
    semantic verification that every claim is entailed by its
    cited text.
    """
    gold_facts = parse_supporting_facts(supporting_facts)

    evidence_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in evidence_chunks
        if chunk.get("chunk_id")
    }

    citations = [
        source_id
        for finding in answer.findings
        for source_id in finding.source_ids
    ]

    if not gold_facts:
        raise ValueError(
            "Citation evaluation requires supporting facts."
        )

    if not citations:
        return {
            "citation_resolution_rate": 0.0,
            "gold_fact_citation_precision": 0.0,
            "gold_fact_citation_recall": 0.0,
        }

    resolved_count = 0
    supported_citation_count = 0
    covered_gold_facts = set()

    for source_id in citations:
        chunk = evidence_by_id.get(source_id)

        if chunk is None:
            continue

        resolved_count += 1

        supported_facts = facts_recovered_by_chunk(
            chunk,
            str(example_id),
            gold_facts,
        )

        if supported_facts:
            supported_citation_count += 1
            covered_gold_facts.update(supported_facts)

    return {
        "citation_resolution_rate": (
            resolved_count / len(citations)
        ),
        "gold_fact_citation_precision": (
            supported_citation_count / len(citations)
        ),
        "gold_fact_citation_recall": (
            len(covered_gold_facts) / len(gold_facts)
        ),
    }
