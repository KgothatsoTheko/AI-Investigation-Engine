# AI Usage Report — AI Investigation Engine

**Report date:** 29 October 2026

## 1. Purpose of AI use

AI tools were used as development assistance during the implementation and debugging of the AI Investigation Engine. The project remains a software system whose behavior is determined by the checked-in Python code, retrieval/indexing components, configured embedding model, data, and Gemini API calls—not by this report.

## 2. AI-assisted development

ChatGPT was used to assist with:
- breaking the assignment requirements into implementation tasks;
- discussing the ingestion, chunking, indexing, hybrid retrieval, RAG, evaluation, and Streamlit UI architecture;
- drafting and revising Python implementations and tests;
- interpreting terminal output, debugging failures, and suggesting targeted corrections;
- explaining evaluation metrics and helping structure the evaluation and usage documentation.

The developer ran the project locally, supplied code and execution output for troubleshooting, integrated changes, and ran the test suite.

## 3. AI use inside the application

The application uses the following AI/ML components:

| Component | Role |
|---|---|
| Sentence Transformers `all-MiniLM-L6-v2` | Produces vector embeddings for indexed text and user queries. |
| Gemini API | Generates structured, evidence-grounded answers and supports the planner/tool-calling flow for multi-step investigations. |
| Hybrid retriever | Combines semantic search and SQLite BM25 keyword search with Reciprocal Rank Fusion; ranking and fusion logic are implemented in application code. |

Gemini receives the user's question and retrieved evidence passages as context for answer generation. Users should only upload documents they are authorized to process and share with the configured API provider. The Gemini key is configured outside source control through environment/secrets settings; secrets should never be committed to the repository.

## 4. Human review and verification

Verification has included:
- implementing and running automated unit/integration tests;
- reviewing the retrieved passages and citations in the UI;
- running the answer-evaluation pipeline against 40 questions;
- checking syntax with `python -m py_compile`;
- running `python -m pytest -q`, which completed with **66 passed** after an indentation issue was corrected.

The 66-test result is evidence that the tested local code passed the available automated suite at that point. It does not by itself establish that the currently deployed cloud app has been tested end-to-end or that every assignment requirement has been met.

## 5. Known AI-assisted error and response

During code iteration, an assistant-provided update placed the `retrieve()` and `close()` method definitions at the wrong indentation level. Python compilation and pytest collection exposed the error. The file was corrected, then `py_compile` and the full test suite passed locally.

This demonstrates why generated suggestions were checked with executable validation rather than accepted without review.

## 6. Evaluation-related caveats

- Gemini API quota/rate-limit errors occurred during the single-pass versus multi-step comparison. The saved comparison run contains three error records, so it is not valid to claim that one mode performed better based on that run.
- The 40-question answer evaluation used a primary model and a fallback model. Its aggregate results are not a controlled comparison of those models.
- Citation-ID validation confirms that citations refer to provided evidence IDs, but it does not independently prove semantic entailment.

## 7. Responsible use and limitations

AI assistance can produce incorrect code, misleading explanations, or evaluation conclusions that exceed the evidence. Human review, tests, trace inspection, and explicit recording of failed API runs were used to reduce these risks. Remaining limits—especially cloud corpus availability, persistence, and the incomplete answer-level configuration comparison—should not be represented as solved.

## 8. Disclosure statement

This report describes AI-assisted implementation, debugging, and application inference based on the development record available for this project. Before submission, the developer should confirm that this wording accurately reflects their use of tools and adjust it to match any specific disclosure format required by the course or organization.
