# AI Usage Report — AI Investigation Engine

**Report date:** 29 October 2026

## 1. Purpose of AI Use

AI tools were used as development assistance during the design, implementation, testing, and debugging of the AI Investigation Engine.

The application itself is a software system whose behavior is determined by its Python implementation, retrieval and indexing components, configured embedding model, stored data, external services, and Gemini API integration.

AI assistance supported development but did not replace implementation, testing, or human verification.

## 2. AI-Assisted Development

ChatGPT was used throughout development to assist with:

- Breaking the assignment requirements into manageable implementation tasks.
- Discussing the ingestion, document processing, chunking, indexing, hybrid retrieval, RAG, and investigation architecture.
- Drafting and revising Python implementations and automated tests.
- Explaining database integration, API connections, and deployment configuration.
- Interpreting terminal output and diagnosing implementation errors.
- Suggesting targeted code corrections and improvements.
- Explaining evaluation metrics and helping structure the Evaluation Report and project documentation.

The developer remained responsible for running the application, reviewing the generated code, integrating changes, configuring external services, and verifying implementation results.

## 3. AI Use Inside the Application

The application uses the following AI and machine-learning components:

| Component | Role |
|---|---|
| Sentence Transformers `all-MiniLM-L6-v2` | Generates embeddings for document chunks and user queries to support semantic retrieval. |
| Google Gemini API | Generates structured, evidence-grounded answers and supports the tool-calling workflow used in multi-step investigations. |
| Hybrid retriever | Combines semantic and keyword search results using Reciprocal Rank Fusion (RRF), with ranking and fusion logic implemented in application code. |
| Qdrant Cloud | Stores and searches semantic vector embeddings in the hosted deployment. |
| Supabase PostgreSQL | Stores uploaded-document text and metadata and supports hosted keyword retrieval through PostgreSQL full-text search. |

The production retrieval workflow combines:

1. Semantic retrieval from Qdrant Cloud.
2. Keyword retrieval from Supabase PostgreSQL.
3. Reciprocal Rank Fusion to combine the available rankings.
4. Evidence selection and duplicate filtering.
5. Gemini-based answer generation using retrieved evidence.

The application also supports local SQLite BM25 and local Qdrant implementations for development and evaluation.

Gemini receives the user's question and retrieved evidence passages as context for answer generation. In multi-step mode, the model can use retrieval tools to conduct additional targeted searches before producing its final response.

Users should only upload documents they are authorized to process and share with the configured API provider. API keys and database credentials are kept outside source control and configured through environment variables or Streamlit Community Cloud Secrets.

## 4. Human Review and Verification

Human review and technical verification were performed throughout development.

Verification activities included:

- Reviewing document extraction results and metadata.
- Inspecting retrieved evidence passages and generated citations in the application.
- Running the answer-evaluation pipeline against 40 questions.
- Checking Python syntax using `python -m py_compile`.
- Running automated unit and integration tests.
- Inspecting errors and stack traces when tests failed.
- Reviewing changes before committing and deploying them.
- Verifying Qdrant Cloud and PostgreSQL migration counts.
- Checking the deployed Streamlit application and its search-source status.

### Automated Testing

The latest recorded local test run on 30 October 2026 completed successfully:

**68 tests passed.**

The test suite covers implemented areas including:

- Document storage and management.
- Document upload and indexing.
- Semantic and keyword retrieval.
- Index maintenance and deletion.
- Investigation behavior.
- Qdrant local/cloud isolation.

The final test runs were performed with cloud connection environment variables cleared so that tests used isolated local databases.

### Production Verification

The hosted deployment was also checked independently.

The Streamlit application displayed:

| Production verification | Result |
|---|---:|
| Main corpus vector chunks | 67,507 |
| Main corpus keyword records | 67,507 |
| Qdrant Cloud endpoint | Configured |
| Hosted PostgreSQL storage | Configured |

The Qdrant migration was independently verified with 67,507 vectors in the remote collection, using 384-dimensional embeddings.

The PostgreSQL migration reported 67,507 imported keyword records and 67,507 records in the main keyword index.

These results establish that the main research indexes are accessible to the deployed application. They do not, by themselves, establish that every document-management operation has passed a complete production end-to-end test.

## 5. Known AI-Assisted Errors and Responses

AI-generated code suggestions were not always correct. Several issues were identified through execution and testing.

### Incorrect Python indentation

During code iteration, an assistant-provided update placed the `retrieve()` and `close()` method definitions at the wrong indentation level.

Python compilation and pytest collection exposed the problem. The file was corrected, after which compilation and the automated test suite passed.

### PostgreSQL database API error

An earlier implementation attempted to call `executemany()` directly on a Psycopg connection instead of using a cursor.

The resulting error was identified from the migration traceback. The implementation was corrected to use the appropriate cursor operation, and the subsequent PostgreSQL migration completed successfully.

### Qdrant cloud and test isolation issue

During testing, an implementation allowed a Qdrant client to select the remote cloud service even when a test supplied a temporary local database path.

As a result, automated tests interacted with a production collection and caused the main research vector collection to become empty.

The issue was identified by comparing the remote Qdrant count against the local source database.

The local source still contained 67,507 vectors. The production collection was recreated and restored from that verified source, and the remote vector count was checked again.

The Qdrant implementation was then updated to separate temporary test databases from the application's hosted database connection. Additional regression tests were added to verify this separation.

The final local test suite completed with 68 passing tests.

These incidents demonstrate the importance of examining generated code, executing tests, inspecting actual database behavior, and verifying production data rather than assuming an AI-generated solution is correct.

## 6. Evaluation-Related Caveats

The evaluation process also encountered limitations:

- Gemini API quota and rate-limit errors interrupted the single-pass versus multi-step comparison. The saved comparison contains three error records and does not support a valid performance conclusion between the two modes.
- The 40-question answer evaluation used a primary model and a fallback model. Its aggregate results are therefore not a controlled comparison between those models.
- Citation-ID validation confirms that citations resolve to retrieved evidence identifiers. It does not independently establish that every generated claim is semantically supported.
- The reported answer metrics show room for improvement in answer correctness and supporting-evidence coverage.

These limitations are documented in the Evaluation Report rather than being presented as successful experiments.

## 7. Responsible Use and Limitations

AI-assisted development can introduce incorrect implementations, incomplete reasoning, misleading explanations, or evaluation conclusions that exceed the available evidence.

The following practices were used to reduce these risks:

- Reviewing and integrating generated code rather than accepting it without inspection.
- Running syntax checks and automated tests after implementation changes.
- Examining error messages and stack traces to diagnose failures.
- Inspecting retrieved evidence and generated citations.
- Verifying migrated data using actual database counts.
- Keeping external credentials outside source control.
- Documenting incomplete experiments and known limitations.

Remaining limitations include the incomplete single-pass versus multi-step answer comparison, mixed-model evaluation, and the need for broader evaluation of unanswerable and contradictory questions.

## 8. Disclosure Statement

ChatGPT was used as a development assistant for architecture discussions, implementation guidance, code drafting, debugging, test interpretation, evaluation analysis, and documentation.

The developer implemented and integrated the resulting changes, configured the external services, executed the application and tests, reviewed outputs, and verified the deployment and migrated data.

AI-generated outputs were treated as suggestions requiring technical review and validation.

This report describes the AI assistance and verification activities undertaken during development of the AI Investigation Engine.