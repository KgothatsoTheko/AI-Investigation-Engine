# Evaluation Report — AI Investigation Engine

**Evaluation date:** 29 October 2026  
**Status:** Partial evaluation; limitations are documented below.

## 1. Objective

This evaluation examines whether the AI Investigation Engine retrieves useful evidence, produces answers that match reference answers, and cites evidence identifiers that exist in the retrieved passages. It also records the current comparison of the two chunking strategies.

## 2. Corpus and retrieval setup

The research corpus was built from the `xanhho/2WikiMultihopQA` dataset at revision `6ef4eb1bc94ca4a768bb86809fafe777ad795995`.

| Corpus build measure | Result |
|---|---:|
| Source examples scanned | 10,878 |
| Source passage occurrences | 108,780 |
| Unique text records retained | 50,001 |
| Duplicate occurrences removed | 58,779 |
| Evaluation seed records | 100 |
| Questions used for answer evaluation | 40 |

The retrieval pipeline combines semantic retrieval from Qdrant with keyword retrieval using SQLite BM25. The ranked lists are combined with Reciprocal Rank Fusion (RRF), then duplicate evidence is filtered. The default retriever requests up to 25 candidates and returns up to 12 final evidence chunks for single-pass answer generation.

## 3. Chunking-strategy comparison

Two chunking strategies were built and inspected:

| Measure | Fixed-window | Sentence-aware / sentence-grouped |
|---|---:|---:|
| Chunks generated | 64,744 | 67,507 |
| Average words per chunk | 55.33 | 54.54 |
| Empty chunks found | 0 | 0 |
| Oversized chunks | 0 | 19 |

The sentence-aware approach produces more chunks and a slightly lower average chunk size. It also produced 19 oversized chunks in this run. These are structural measurements; they do **not** establish that one strategy has better answer quality. A controlled answer-level comparison is still needed before claiming a winner.

## 4. Answer-quality results

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

**Interpretation**

- Citation resolution of 1.000 means the cited chunk identifiers resolved against the evidence supplied to the generator. It does not prove that every citation fully entails its associated claim.
- Gold-fact citation precision of 0.925 indicates that most evaluated citations aligned with the reference supporting facts.
- Gold-fact citation recall of 0.6488 indicates that the answer citations covered roughly two-thirds of the reference supporting facts on average.
- Exact match and token F1 of 0.400 show substantial room for improvement in direct-answer correctness.
- The evaluation questions have reference answers and supporting facts. An `insufficient_evidence` response therefore counts as a missed answer on this supported-question benchmark; abstention is valuable for genuinely unanswerable questions, but this benchmark alone does not measure that behavior.

The answers were produced across a primary model and a fallback model after provider errors or quota limits. The aggregate results should therefore be treated as results for the configured pipeline, not as a clean head-to-head model comparison.

## 5. Retrieval coverage and error analysis

Average supporting-evidence retrieval recall was approximately **0.649**, and all gold supporting evidence was retrieved for about **35%** of evaluated questions.

| Question category | Average retrieval recall |
|---|---:|
| Bridge-comparison | 0.458 |
| Comparison | 0.962 |
| Compositional | 0.542 |
| Inference | 0.500 |

The inference category contained only two evaluated questions, so that value is not a reliable estimate for the broader category. Bridge-comparison and compositional questions appear to be the main retrieval weaknesses in this sample. These questions require combining evidence across more than one fact, so missing even one supporting passage can prevent a complete response.

Priority improvements are to increase supporting-passage coverage for multi-hop questions, inspect failed cases individually, and evaluate whether sentence-aware chunking changes retrieval recall rather than comparing chunk counts alone.

## 6. Multi-step versus single-pass comparison

The saved `single_vs_multistep_results.jsonl` run contains three error records associated with Gemini API quota/rate-limit failures. It does not provide a valid answer-quality comparison between the two modes.

No performance winner is claimed for single-pass RAG versus multi-step investigation. To complete this comparison, rerun both modes against the same questions and corpus with working API quota, or run a separate deterministic retrieval-only comparison that does not require answer-generation calls. Report the number of successfully evaluated examples and preserve failed/error rows separately from successful metric rows.

## 7. Limitations and reproducibility

1. **Mixed generation models:** fallback behavior means the 40-question aggregate is not attributable to one model alone.
2. **Small category samples:** in particular, inference has only two examples.
3. **Incomplete mode comparison:** quota errors prevented the saved single-pass versus multi-step experiment from producing usable comparative results.
4. **Deployment corpus availability:** the local build contains 50,001 unique records, but `data/raw/` and `data/processed/` are excluded by `.gitignore`. Unless the corpus indexes are provisioned separately, the public deployment does not contain the full local corpus.
5. **Persistence:** Streamlit Community Cloud does not guarantee persistence of files created during app runtime. The local SQLite and Qdrant upload stores should not be described as durable cloud persistence.
6. **Scope of citation metrics:** citation-ID validation tests whether referenced chunk IDs exist; it is not a full semantic entailment or factuality judge.

## 8. Overall assessment

The project has a working local evaluation pipeline, 40 evaluated answer records, measurable citation metrics, two constructed chunking strategies, and a hybrid semantic/keyword retrieval design with custom RRF ranking. Results show reliable citation-ID resolution but only moderate answer matching and incomplete evidence coverage.

The evaluation should be considered **partially complete** until the two-configuration comparison succeeds and the deployed corpus/persistence limitations are either resolved or explicitly accepted by the assignment rubric. No missing metric is inferred or presented as a successful result.
