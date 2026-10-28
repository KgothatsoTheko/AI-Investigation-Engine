
import json
import re
from typing import Any


def normalize_title(title: Any) -> str:
    """Normalize title spelling and punctuation for comparison."""
    text = str(title).casefold()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def parse_supporting_facts(
    value: Any,
) -> set[tuple[str, int]]:
    """
    Convert supporting facts into normalized (title, sentence index)
    pairs.

    Example:
        [["Mario Bava", 0]]
        becomes:
        {("mariobava", 0)}
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return set()

    if not isinstance(value, (list, tuple)):
        return set()

    facts = set()

    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        title = normalize_title(item[0])

        try:
            sentence_index = int(item[1])
        except (TypeError, ValueError):
            continue

        if title and sentence_index >= 0:
            facts.add((title, sentence_index))

    return facts


def facts_recovered_by_chunk(
    chunk: dict[str, Any],
    example_id: str,
    gold_facts: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    """
    Identify which of a question's labelled supporting facts
    are covered by this chunk.

    Require matching title, source sentence index and provenance
    from the specific evaluation example.
    """
    chunk_title = normalize_title(chunk.get("title", ""))

    if not chunk_title:
        return set()

    chunk_sentence_indices = set()

    for value in chunk.get("source_sentence_indices", []):
        try:
            chunk_sentence_indices.add(int(value))
        except (TypeError, ValueError):
            continue

    # Confirm that this passage was part of the particular
    # example being evaluated, not merely a similarly titled record.
    provenance_indices = set()

    for provenance in chunk.get("provenance", []):
        if provenance.get("example_id") != example_id:
            continue

        for value in provenance.get(
            "supporting_sentence_indices", []
        ):
            try:
                provenance_indices.add(int(value))
            except (TypeError, ValueError):
                continue

    available_indices = (
        chunk_sentence_indices & provenance_indices
    )

    return {
        (title, sentence_index)
        for title, sentence_index in gold_facts
        if title == chunk_title
        and sentence_index in available_indices
    }


def evaluate_ranked_results(
    results: list[dict[str, Any]],
    example_id: str,
    gold_facts: set[tuple[str, int]],
    recall_cutoffs: tuple[int, ...] = (5, 10),
    mrr_cutoff: int = 10,
) -> dict[str, Any]:
    """
    Compute evidence recall, MRR and complete evidence coverage.

    Each result must contain a 'chunk' dictionary with its source
    title, sentence indices and provenance.
    """
    if not gold_facts:
        raise ValueError(
            "Cannot evaluate retrieval without supporting facts."
        )

    if any(k <= 0 for k in recall_cutoffs):
        raise ValueError("Recall cutoffs must be positive.")

    if mrr_cutoff <= 0:
        raise ValueError("mrr_cutoff must be positive.")

    result_hits = []

    for result in results:
        chunk = result.get("chunk", {})

        result_hits.append(
            facts_recovered_by_chunk(
                chunk,
                example_id,
                gold_facts,
            )
        )

    metrics: dict[str, Any] = {
        "gold_fact_count": len(gold_facts),
    }

    for cutoff in recall_cutoffs:
        covered = set()

        for hits in result_hits[:cutoff]:
            covered.update(hits)

        metrics[f"evidence_recall@{cutoff}"] = (
            len(covered) / len(gold_facts)
        )

        metrics[f"supporting_facts_found@{cutoff}"] = (
            len(covered)
        )

    reciprocal_rank = 0.0

    for rank, hits in enumerate(
        result_hits[:mrr_cutoff],
        start=1,
    ):
        if hits:
            reciprocal_rank = 1.0 / rank
            break

    metrics[f"mrr@{mrr_cutoff}"] = reciprocal_rank

    max_cutoff = max(recall_cutoffs)
    max_recall = metrics[f"evidence_recall@{max_cutoff}"]

    metrics[f"complete_evidence@{max_cutoff}"] = float(
        max_recall == 1.0
    )

    return metrics
