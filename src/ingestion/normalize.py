
import hashlib
import json
import re
from typing import Any


DATASET_NAME = "2WikiMultihopQA"


def parse_json_field(value: Any) -> Any:
    """Decode a JSON string when necessary."""
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_whitespace(value: Any) -> str:
    """Collapse repeated spaces and remove surrounding whitespace."""
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def content_to_text(content: Any) -> str:
    """Convert a passage's content into one clean text string."""
    if isinstance(content, (list, tuple)):
        parts = [
            normalize_whitespace(sentence)
            for sentence in content
            if sentence is not None
        ]
        return normalize_whitespace(" ".join(parts))

    return normalize_whitespace(content)


def supporting_indices_by_title(
    supporting_facts: Any,
) -> dict[str, list[int]]:
    """
    Map each passage title to its supporting sentence indices.

    Example:
        [["Stuart Rosenberg", 0]]
        becomes:
        {"stuart rosenberg": [0]}
    """
    facts = parse_json_field(supporting_facts)
    indices: dict[str, set[int]] = {}

    if not isinstance(facts, (list, tuple)):
        return {}

    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) < 2:
            continue

        title = normalize_whitespace(fact[0]).casefold()

        try:
            sentence_index = int(fact[1])
        except (TypeError, ValueError):
            continue

        if title:
            indices.setdefault(title, set()).add(sentence_index)

    return {
        title: sorted(sentence_indices)
        for title, sentence_indices in indices.items()
    }


def example_to_documents(
    example: dict[str, Any],
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Convert one QA example into individual source passage records.

    Questions and expected answers are deliberately excluded
    from these searchable source records.
    """
    example_id = example.get("_id", example.get("id"))

    if example_id is None:
        raise ValueError("Dataset example is missing its ID.")

    example_id = normalize_whitespace(example_id)

    context = parse_json_field(example.get("context", []))

    if not isinstance(context, (list, tuple)):
        raise ValueError(
            f"Context for example {example_id} is not a list."
        )

    supporting_indices = supporting_indices_by_title(
        example.get("supporting_facts", [])
    )

    documents = []

    for passage_index, passage in enumerate(context):
        if isinstance(passage, dict):
            title = passage.get("title", "")
            content = passage.get(
                "content",
                passage.get("sentences", passage.get("text", "")),
            )

        elif isinstance(passage, (list, tuple)) and len(passage) >= 2:
            title, content = passage[0], passage[1]

        else:
            raise ValueError(
                f"Unexpected passage format in example {example_id}, "
                f"passage index {passage_index}."
            )

        title = normalize_whitespace(title)
        text = content_to_text(content)

        if not title or not text:
            raise ValueError(
                f"Empty title or text in example {example_id}, "
                f"passage index {passage_index}."
            )

        # Generate a deterministic ID from the passage's content.
        # Identical title/text pairs receive the same ID across examples.
        identity = f"{title.casefold()}\n{text}"

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]

        document_id = f"2wiki_{digest}"

        documents.append(
            {
                "document_id": document_id,
                "title": title,
                "text": text,
                "metadata": {
                    "dataset": DATASET_NAME,
                    "split": split,
                },
                "provenance": [
                    {
                        "example_id": example_id,
                        "passage_index": passage_index,
                        "supporting_sentence_indices": (
                            supporting_indices.get(
                                title.casefold(), []
                            )
                        ),
                    }
                ],
            }
        )

    return documents


def example_to_evaluation_item(
    example: dict[str, Any],
) -> dict[str, Any]:
    """Keep QA answers and evidence labels separate from source text."""
    example_id = example.get("_id", example.get("id"))

    if example_id is None:
        raise ValueError("Dataset example is missing its ID.")

    return {
        "example_id": normalize_whitespace(example_id),
        "question": normalize_whitespace(
            example.get("question", "")
        ),
        "answer": normalize_whitespace(
            example.get("answer", "")
        ),
        "question_type": example.get("type"),
        "supporting_facts": parse_json_field(
            example.get("supporting_facts", [])
        ),
        "evidences": parse_json_field(
            example.get("evidences", [])
        ),
    }
