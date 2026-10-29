from types import SimpleNamespace

from src.rag.retriever import HybridEvidenceRetriever


class FakeEncoder:
    def encode_query(self, question):
        return [1.0, 0.0, 0.0]


class FakeKeywordIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=20):
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk": chunk,
                "score": 2.0,
                "rank": 1,
                "matched_terms": ["evidence"],
            }
            for chunk in self.chunks[:top_k]
        ]


class FakeVectorStore:
    def __init__(self, chunks, collection_exists=True):
        self.chunks = chunks
        self.collection_exists = collection_exists

    def has_collection(self):
        return self.collection_exists

    def search(self, query_vector, limit=5):
        return [
            SimpleNamespace(
                payload=chunk,
                score=0.9,
            )
            for chunk in self.chunks[:limit]
        ]


def make_chunk(chunk_id, document_id, title, text):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": title,
        "text": text,
        "metadata": {},
        "provenance": [],
    }


def make_retriever(main_chunk, upload_chunk):
    retriever = HybridEvidenceRetriever.__new__(
        HybridEvidenceRetriever
    )

    retriever.candidate_limit = 10
    retriever.final_limit = 10
    retriever.encoder = FakeEncoder()

    retriever.keyword_index = FakeKeywordIndex(
        [main_chunk]
    )
    retriever.vector_store = FakeVectorStore(
        [main_chunk]
    )

    retriever.upload_keyword_index = FakeKeywordIndex(
        [upload_chunk]
    )
    retriever.upload_vector_store = FakeVectorStore(
        [upload_chunk]
    )

    return retriever


def test_retriever_returns_evidence_from_main_and_upload_indexes():
    main_chunk = make_chunk(
        "main_chunk_1",
        "main_document",
        "Main corpus document",
        "Main corpus evidence about investigation systems.",
    )

    upload_chunk = make_chunk(
        "upload_chunk_1",
        "upload_task",
        "Uploaded task document",
        "Uploaded evidence about investigation systems.",
    )

    retriever = make_retriever(
        main_chunk,
        upload_chunk,
    )

    results = retriever.retrieve(
        "What evidence describes investigation systems?"
    )

    document_ids = {
        result["chunk"]["document_id"]
        for result in results
    }

    assert "main_document" in document_ids
    assert "upload_task" in document_ids


def test_retriever_works_without_upload_indexes():
    main_chunk = make_chunk(
        "main_chunk_1",
        "main_document",
        "Main corpus document",
        "Main corpus evidence about investigation systems.",
    )

    retriever = make_retriever(
        main_chunk,
        make_chunk(
            "upload_chunk_1",
            "upload_task",
            "Uploaded task document",
            "Uploaded evidence about investigation systems.",
        ),
    )

    retriever.upload_keyword_index = None
    retriever.upload_vector_store = None

    results = retriever.retrieve("investigation systems")

    assert results
    assert all(
        result["chunk"]["document_id"] == "main_document"
        for result in results
    )