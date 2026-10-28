
import math
import re
from collections import Counter, defaultdict
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """
    A small, transparent BM25 implementation.

    This version keeps its inverted index in memory. We will use
    it on the 500-chunk prototype before deciding how best to
    scale keyword retrieval to the complete corpus.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.documents = documents
        self.k1 = k1
        self.b = b
        self.document_count = len(documents)

        # term -> {document index: frequency in that document}
        self.inverted_index: dict[
            str, dict[int, int]
        ] = defaultdict(dict)

        self.document_lengths: list[int] = []

        for document_index, document in enumerate(documents):
            # Titles matter for exact names and film titles.
            searchable_text = (
                f"{document.get('title', '')} "
                f"{document.get('text', '')}"
            )

            term_frequencies = Counter(
                tokenize(searchable_text)
            )

            document_length = sum(
                term_frequencies.values()
            )

            self.document_lengths.append(document_length)

            for term, frequency in term_frequencies.items():
                self.inverted_index[term][
                    document_index
                ] = frequency

        total_words = sum(self.document_lengths)

        self.average_document_length = (
            total_words / self.document_count
            if self.document_count
            else 0.0
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring keyword matches."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = list(dict.fromkeys(tokenize(query)))

        if not query_terms or not self.documents:
            return []

        if self.average_document_length == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        matched_terms: dict[int, set[str]] = defaultdict(set)

        for term in query_terms:
            postings = self.inverted_index.get(term)

            if not postings:
                continue

            # Number of documents containing this term.
            document_frequency = len(postings)

            # BM25 inverse document frequency.
            inverse_document_frequency = math.log(
                1
                + (
                    self.document_count
                    - document_frequency
                    + 0.5
                )
                / (document_frequency + 0.5)
            )

            for document_index, term_frequency in postings.items():
                document_length = self.document_lengths[
                    document_index
                ]

                length_adjustment = self.k1 * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )

                term_score = (
                    inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + length_adjustment)
                )

                scores[document_index] += term_score
                matched_terms[document_index].add(term)

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                self.documents[item[0]].get("chunk_id", ""),
            ),
        )

        results = []

        for rank, (document_index, score) in enumerate(
            ranked_documents[:top_k],
            start=1,
        ):
            document = self.documents[document_index]

            results.append(
                {
                    "chunk_id": document.get("chunk_id"),
                    "chunk": document,
                    "score": float(score),
                    "rank": rank,
                    "matched_terms": sorted(
                        matched_terms[document_index]
                    ),
                }
            )

        return results
