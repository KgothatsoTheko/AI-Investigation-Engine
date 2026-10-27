
import hashlib
import re
from typing import Any


def _get_sentences(document: dict[str, Any]) -> list[str]:
    """
    Prefer preserved source sentences.

    The fallback supports older records or manually supplied
    documents that only contain a text field.
    """
    sentences = document.get("sentences")

    if isinstance(sentences, list) and sentences:
        return [
            str(sentence).strip()
            for sentence in sentences
            if str(sentence).strip()
        ]

    text = str(document.get("text", "")).strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def fixed_window_chunks(
    sentences: list[str],
    chunk_size_words: int = 100,
    overlap_words: int = 20,
) -> list[dict[str, Any]]:
    """
    Split passage text into fixed-size overlapping word windows.

    Word positions use an end-exclusive range:
    start_word=0, end_word=100 means words 0 through 99.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")

    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be >= 0 and smaller "
            "than chunk_size_words."
        )

    words = []
    sentence_ranges = []

    for sentence_index, sentence in enumerate(sentences):
        sentence_words = sentence.split()

        if not sentence_words:
            continue

        start_word = len(words)
        words.extend(sentence_words)
        end_word = len(words)

        sentence_ranges.append(
            (sentence_index, start_word, end_word)
        )

    chunks = []
    start = 0
    step = chunk_size_words - overlap_words

    while start < len(words):
        end = min(start + chunk_size_words, len(words))

        source_sentence_indices = [
            sentence_index
            for sentence_index, sentence_start, sentence_end
            in sentence_ranges
            if sentence_start < end and sentence_end > start
        ]

        chunks.append(
            {
                "text": " ".join(words[start:end]),
                "start_word": start,
                "end_word": end,
                "source_sentence_indices": (
                    source_sentence_indices
                ),
            }
        )

        if end == len(words):
            break

        start += step

    return chunks


def sentence_group_chunks(
    sentences: list[str],
    max_words: int = 100,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """
    Group complete source sentences up to a target word count.

    If a single sentence exceeds max_words, it remains intact.
    Overlap reuses the last sentence of a chunk in the next chunk
    where possible.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative.")

    indexed_sentences = [
        (index, str(sentence).strip())
        for index, sentence in enumerate(sentences)
        if str(sentence).strip()
    ]

    chunks = []
    start = 0

    while start < len(indexed_sentences):
        selected = []
        word_count = 0
        end = start

        while end < len(indexed_sentences):
            sentence_index, sentence = indexed_sentences[end]
            sentence_word_count = len(sentence.split())

            # Always include at least one sentence, even if it
            # exceeds the target size.
            if selected and (
                word_count + sentence_word_count > max_words
            ):
                break

            selected.append((sentence_index, sentence))
            word_count += sentence_word_count
            end += 1

        chunks.append(
            {
                "text": " ".join(
                    sentence for _, sentence in selected
                ),
                "start_word": None,
                "end_word": None,
                "source_sentence_indices": [
                    sentence_index
                    for sentence_index, _ in selected
                ],
            }
        )

        if end >= len(indexed_sentences):
            break

        # Guarantee forward progress even when one sentence
        # is longer than the configured maximum.
        start = max(
            start + 1,
            end - overlap_sentences,
        )

    return chunks


def chunk_document(
    document: dict[str, Any],
    strategy: str,
    chunk_size_words: int = 100,
    overlap_words: int = 20,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """
    Apply one chunking strategy to a source document.

    Each output record preserves source identity and provenance.
    """
    document_id = document.get("document_id")

    if not document_id:
        raise ValueError("Document is missing document_id.")

    sentences = _get_sentences(document)

    if not sentences:
        return []

    if strategy == "fixed_window":
        raw_chunks = fixed_window_chunks(
            sentences,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

    elif strategy == "sentence_grouped":
        raw_chunks = sentence_group_chunks(
            sentences,
            max_words=chunk_size_words,
            overlap_sentences=overlap_sentences,
        )

    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )

    result = []

    for chunk_index, raw_chunk in enumerate(raw_chunks):
        chunk_text = raw_chunk["text"]

        identity = (
            f"{document_id}\n"
            f"{strategy}\n"
            f"{chunk_index}\n"
            f"{chunk_text}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]

        result.append(
            {
                "chunk_id": f"chunk_{digest}",
                "document_id": document_id,
                "title": document.get("title", ""),
                "text": chunk_text,
                "metadata": document.get("metadata", {}),
                "provenance": document.get("provenance", []),
                "source_sentence_indices": (
                    raw_chunk["source_sentence_indices"]
                ),
                "chunking": {
                    "strategy": strategy,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_text.split()),
                    "start_word": raw_chunk["start_word"],
                    "end_word": raw_chunk["end_word"],
                },
            }
        )

    return result
