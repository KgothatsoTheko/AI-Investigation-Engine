# Evaluation Report — AI Investigation Engine

**Evaluation date:** 29 October 2026  
**Status:** Partial evaluation; implementation and deployment are verified, while some answer-quality comparisons remain incomplete.

## 1. Objective

This evaluation examines whether the AI Investigation Engine retrieves useful evidence, produces answers that match reference answers, and cites evidence identifiers that resolve to retrieved passages. It also records the comparison of two chunking strategies and verifies the deployed search infrastructure.

## 2. Corpus and Retrieval Setup

The research corpus was built from the `xanhho/2WikiMultihopQA` dataset at revision `6ef4eb1bc94ca4a768bb86809fafe777ad795995`.

| Corpus build measure | Result |
|---|---:|
| Source examples scanned | 10,878 |
| Source passage occurrences | 108,780 |
| Unique text records retained | 50,001 |
| Duplicate occurrences removed | 58,779 |
| Evaluation seed records | 100 |
| Questions used for answer evaluation | 40 |
| Sentence-grouped chunks generated | 67,507 |

The retrieval pipeline combines semantic retrieval from Qdrant with keyword retrieval. The ranked lists are combined using Reciprocal Rank Fusion (RRF), followed by duplicate-evidence filtering.

The default retriever requests up to 25 candidates and returns up to 12 final evidence chunks for single-pass answer generation.

**Hosted retrieval implementation:**

- Qdrant Cloud stores the semantic vectors.
- Supabase PostgreSQL stores the keyword-search records and performs PostgreSQL full-text search.
- Local development and evaluation also support SQLite BM25 and local Qdrant.
- Reciprocal Rank Fusion combines the available semantic and keyword rankings.

## 3. Chunking-Strategy Comparison

Two chunking strategies were built and inspected:

| Measure | Fixed-window | Sentence-aware / sentence-grouped |
|---|---:|---:|
| Chunks generated | 64,744 | 67,507 |
| Average words per chunk | 55.33 | 54.54 |
| Empty chunks found | 0 | 0 |
| Oversized chunks | 0 | 19 |

The sentence-aware approach produces more chunks and a slightly lower average chunk size. It also produced 19 oversized chunks in this run.

These are structural measurements; they do not establish that one strategy has better answer quality. A controlled answer-level comparison is still needed before claiming a winner.

## 4. Answer-Quality Results

The current 40-question answer evaluation produced the following aggregate metrics:

| Metric | Result |
|---|---:|
| Exact match | 0.400 |
| Token F1 | 0.400 |
| Citation resolution rate | 1.000 |
| Gold-fact citation precision | 0.925 |
| Gold-fact citation recall | 0.6488 |
| Questions answered | 18 / 40 (45%) |
| Responses marked insufficient evidence | 22 / 40 (55%) |

### Interpretation

- Citation resolution of 1.000 means the cited chunk identifiers resolved against the evidence supplied to the generator. It does not prove that every citation fully entails its associated claim.
- Gold-fact citation precision of 0.925 indicates that most evaluated citations aligned with the reference supporting facts.
- Gold-fact citation recall of 0.6488 indicates that answer citations covered roughly two-thirds of the reference supporting facts on average.
- Exact match and token F1 of 0.400 show substantial room for improvement in direct-answer correctness.
- The evaluation questions have reference answers and supporting facts. An `insufficient_evidence` response therefore counts as a missed answer on this supported-question benchmark. Abstention is valuable for genuinely unanswerable questions, but this benchmark alone does not measure that behavior.

The answers were produced across a primary model and a fallback model after provider errors or quota limits. The aggregate results should therefore be treated as results for the configured pipeline, not as a clean head-to-head model comparison.

## 5. Retrieval Coverage and Error Analysis

Average supporting-evidence retrieval recall was approximately **0.649**, and all gold supporting evidence was retrieved for about **35%** of evaluated questions.

| Question category | Average retrieval recall |
|---|---:|
| Bridge-comparison | 0.458 |
| Comparison | 0.962 |
| Compositional | 0.542 |
| Inference | 0.500 |

The inference category contained only two evaluated questions, so that value is not a reliable estimate for the broader category.

Bridge-comparison and compositional questions appear to be the main retrieval weaknesses in this sample. These questions require combining evidence across more than one fact, so missing even one supporting passage can prevent a complete response.

Priority improvements are to increase supporting-passage coverage for multi-hop questions, inspect failed cases individually, and evaluate whether sentence-aware chunking changes retrieval recall rather than comparing chunk counts alone.

## 6. Deployed Infrastructure Verification

The application was deployed using Streamlit Community Cloud, with Qdrant Cloud and Supabase PostgreSQL providing hosted storage and retrieval services.

On 30 October 2026, the production application displayed the following status in its sidebar:

| Production component | Verified status |
|---|---|
| Main corpus vector index | Connected |
| Main corpus keyword index | Connected |
| Hosted PostgreSQL storage | Configured |
| Remote Qdrant endpoint | Configured |
| Main corpus vector chunks | 67,507 |
| Main corpus keyword records | 67,507 |

The Qdrant migration was independently verified after copying the local vector database to the remote collection:

| Migration measure | Result |
|---|---:|
| Local source vectors | 67,507 |
| Remote destination vectors | 67,507 |
| Vector dimension | 384 |
| Migration count verification | Passed |

The PostgreSQL migration also reported:

- 67,507 keyword records imported.
- 67,507 records present in the main keyword index.
- Successful PostgreSQL connection and database verification.

The deployed application therefore has access to the populated main-corpus semantic and keyword indexes.

These checks establish that the production application can see the expected index counts. They do not independently measure answer quality or replace a complete production smoke test covering document upload, indexing, retrieval, updating, and deletion.

## 7. Multi-Step Versus Single-Pass Comparison

The saved `single_vs_multistep_results.jsonl` run contains three error records associated with Gemini API quota/rate-limit failures. It does not provide a valid answer-quality comparison between the two modes.

No performance winner is claimed for single-pass RAG versus multi-step investigation.

To complete this comparison, rerun both modes against the same questions and corpus with working API quota, or perform a separate deterministic retrieval-only comparison that does not require answer-generation calls.

Report the number of successfully evaluated examples and preserve failed/error rows separately from successful metric rows.

## 8. Limitations and Reproducibility

1. **Mixed generation models:** Fallback behavior means the 40-question aggregate is not attributable to one model alone.

2. **Small category samples:** In particular, inference has only two evaluated questions.

3. **Incomplete mode comparison:** Quota errors prevented the saved single-pass versus multi-step experiment from producing usable comparative results.

4. **Evaluation coverage:** The reported 40-question metrics do not establish performance across every intended question category. Broader evaluation of unanswerable and contradictory questions remains necessary.

5. **Deployment verification scope:** The production sidebar and migration outputs verify that the main indexes are connected and populated. A complete production upload, re-index, search, and deletion smoke test should also be recorded.

6. **Uploaded-file retention:** The hosted document library stores extracted document text and metadata. Original uploaded PDF, DOCX, and TXT binaries are not currently retained for download.

7. **Scope of citation metrics:** Citation-ID validation tests whether referenced chunk IDs exist in the retrieved evidence. It is not a full semantic entailment or factuality evaluation.

8. **External service dependency:** Production retrieval and persistence depend on Qdrant Cloud, Supabase PostgreSQL, and their availability and configured connection credentials.

## 9. Overall Assessment

The project has a working local evaluation pipeline, 40 evaluated answer records, measurable citation metrics, two constructed chunking strategies, and a deployed hybrid semantic and keyword retrieval architecture.

The main research corpus contains 50,001 unique text records and 67,507 sentence-grouped chunks. Its production search indexes have been migrated and verified:

- 67,507 semantic vectors in Qdrant Cloud.
- 67,507 keyword records in Supabase PostgreSQL.

The deployed Streamlit application confirms that both main-corpus indexes are connected.

The evaluation shows reliable citation-ID resolution but only moderate answer matching and incomplete evidence coverage.

The project should therefore be considered **partially evaluated**, rather than fully validated. The remaining evaluation work includes a successful single-pass versus multi-step comparison, broader testing of unanswerable and contradictory questions, and recording a complete production document-management smoke test.

No missing metric or failed comparison is presented as a successful result.