# Evaluation Report — AI Investigation Engine

## 1. Purpose

This report records the evaluation performed on the AI Investigation Engine during local development. It separates software test results from the evaluation of answer quality. Passing automated tests demonstrates that tested components behave as expected; it does not, by itself, prove that generated answers are accurate.

## 2. Evaluation Environment

- **Operating system:** Microsoft Windows
- **Python:** 3.13
- **Test framework:** Pytest
- **User interface:** Streamlit
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector search:** Qdrant local storage
- **Keyword search:** SQLite-backed BM25
- **LLM integration:** Google Gemini API
- **Main dataset:** [2WikiMultihopQA](https://huggingface.co/datasets/xanhho/2WikiMultihopQA)

## 3. Dataset and Index Preparation

The corpus-building process used the 2WikiMultihopQA dataset. The recorded corpus build reported:

| Measure | Recorded result |
|---|---:|
| Dataset examples scanned | 10,878 |
| Source passages encountered | 108,780 |
| Unique passages retained | 50,001 |
| Duplicate passages excluded | 58,779 |
| Existing Qdrant `investigation_chunks` collection | 67,507 points |

The collection point count and corpus-build count refer to different recorded stages of the project and should not be interpreted as the same measure.

The evaluation workflow was prepared to cover single-step questions, multi-step questions, questions requiring connections between sources, questions whose answers are absent from the evidence, and questions involving contradictory information.

## 4. Automated Software Tests

On 29 October 2026, the local test suite completed with the following result:

| Measure | Result |
|---|---:|
| Pytest tests passed | 64 |
| Tests failed | 0 |
| Test run duration | 11.24 seconds |

Command used:

```bash
python -m pytest -q
```

The test suite included tests for answer-selection stratification, vector-store collection maintenance, vector dimension validation, deletion of obsolete document chunks, SQLite BM25 deletion and replacement, uploaded-document indexing, re-indexing, and document deletion. The result applies to the test suite and code state at the time of that run.

## 5. Functional Smoke Test: Uploaded PDF

The document-upload and search-index workflow was tested locally with the assignment PDF, `AI_Investigation_Engine_Task_English.pdf`.

| Check | Observed result |
|---|---|
| Text extraction | Successful |
| Extracted text length | 8,404 characters |
| Chunks indexed | 15 |
| Vectors upserted | 15 |
| Keyword chunks added | 15 |

A Single-pass RAG query asked: “What are the mandatory requirements for the AI Investigation Engine, and what must be submitted at the end?” After correcting the UI's handling of retrieval results, the displayed answer included the key requirement categories and the five stated deliverables: GitHub Repository, Live Demo, README, Evaluation Report, and AI Usage Report. The retrieved passages displayed source text and chunk/document identifiers.

This is a qualitative smoke test using one uploaded document and one question. It is not a statistically meaningful measure of answer accuracy.

## 6. Automated Answer-Quality Evaluation

The answer-evaluation command was attempted with a limit of 40 questions:

```bash
python -m scripts.evaluate_answers --limit 40
```

The run encountered Gemini API quota/rate-limit errors, including HTTP 429 / `RESOURCE_EXHAUSTED` responses. Attempts to use the configured fallback model were also affected by rate limits. Consequently, a complete 40-question result was not obtained from that run.

For this reason, this report does **not** claim an overall answer-accuracy percentage, retrieval precision/recall score, citation accuracy score, or abstention score. Those metrics should only be reported after a complete evaluation run has finished and its outputs have been reviewed.

## 7. Evaluation Summary

| Area | Status | Evidence |
|---|---|---|
| Automated software tests | Passed for the recorded run | 64 tests passed |
| Uploaded PDF text extraction | Passed in smoke test | 8,404 characters extracted |
| Uploaded-document indexing | Passed in smoke test | 15 chunks, 15 vectors, 15 keyword entries |
| Single-pass question answering | Basic qualitative check passed | One question returned an answer with relevant source passages |
| Multi-step investigation | Not formally scored in the recorded results | Requires a completed, reviewed test run |
| Batch answer-quality evaluation | Inconclusive | 40-question run interrupted by API quota/rate limits |
| Public deployment and persistence after hosted restarts | Not evaluated here | No deployed environment test is included in these results |

## 8. Limitations and Next Steps

The results support that the tested software components and one local document workflow operated successfully. They do not establish the accuracy of the system across a broad question set.

The next evaluation steps are:

1. Re-run the 40-question answer evaluation when API quota is available and ensure the run completes.
2. Record answer correctness, evidence relevance, citation validity, and appropriate abstention separately.
3. Test questions requiring evidence from multiple documents, questions with missing answers, and contradictory evidence.
4. Run and review the same representative questions in both Single-pass RAG and Multi-step investigation.
5. Verify the public deployment URL, database connectivity, and data persistence across application restarts before reporting deployment as complete.
6. Monitor performance with the current local Qdrant collection, which produced a warning because it contains more than 20,000 points.

## 9. Conclusion

The recorded automated test run passed all 64 tests, and a local smoke test successfully extracted and indexed an uploaded PDF and returned a source-backed Single-pass RAG answer. A complete quantitative answer-quality evaluation has not yet been established because the attempted 40-question run was interrupted by API quota/rate limits. The system should therefore be considered functionally tested in the recorded areas, with broader answer-quality and deployment validation still required.
