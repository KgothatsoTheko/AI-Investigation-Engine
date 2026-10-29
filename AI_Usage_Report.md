# AI Usage Report — AI Investigation Engine

## 1. Purpose of This Report

This report explains how AI tools were used during the design, implementation, debugging, testing, and documentation of the AI Investigation Engine. AI assistance was used as a development aid; the project still required local implementation, review, testing, and technical decisions by the developer.

## 2. AI Tools and Models Used

### ChatGPT (OpenAI)

ChatGPT was used as a software-development assistant during the project. Assistance included discussing the system architecture, proposing implementation approaches, explaining errors, suggesting code changes, drafting test cases, interpreting test output, and preparing project documentation.

Examples of work for which ChatGPT provided assistance:

- Structuring the document-ingestion, indexing, retrieval, and answer-generation workflow.
- Working through hybrid retrieval using semantic search, keyword search, and reciprocal rank fusion.
- Debugging Python environment and dependency issues, syntax/indentation errors, and test-collection failures.
- Investigating why retrieval results appeared as empty source passages in the Single-pass RAG interface and suggesting that the UI pass the underlying chunk dictionaries to the answer generator.
- Improving the usability and organisation of the Streamlit interface.
- Drafting the README, Evaluation Report, and this AI Usage Report.

The suggestions were based on code, logs, and outputs shared during development. They were not a substitute for running the application and verifying the changes locally.

### Google Gemini API

The application integrates the Google Gemini API for LLM-based answer generation. The multi-step investigation mode also uses a model-driven tool-calling workflow to request targeted searches against the local retrieval system before generating a final answer.

Gemini was therefore part of the implemented application's runtime, rather than only a tool used to write the code. API quota and rate-limit errors affected an attempted batch answer-evaluation run; that run did not produce a complete 40-question evaluation result.

### Sentence Transformers

The project uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` embedding model to convert text into vectors for semantic search. The generated vectors are stored and searched using Qdrant. This is a model component of the application, not evidence that the application's generated answers are automatically correct.

## 3. How AI Assistance Was Applied

AI assistance was used iteratively across several areas:

1. **Architecture and technical reasoning:** Discussing the roles of document processing, chunking, embeddings, vector search, keyword retrieval, evidence selection, and LLM generation.
2. **Implementation support:** Drafting and refining Python snippets and application logic based on the project's existing modules and requirements.
3. **Debugging:** Interpreting tracebacks and test failures, identifying likely causes, and proposing targeted fixes for the local codebase.
4. **Testing:** Helping interpret Pytest results and propose checks for indexing, re-indexing, deletion, retrieval, and answer selection.
5. **User interface:** Suggesting clearer workflow steps for uploading documents, saving them, indexing them, running investigations, and reviewing evidence.
6. **Documentation:** Preparing written explanations of the architecture, setup instructions, observed test results, limitations, and use of AI tools.

## 4. Developer Contribution and Responsibility

The developer defined the project goal, supplied the task requirements and relevant source code, ran commands in the local Windows environment, inspected application output, and used test results to guide the next changes. The developer installed dependencies, ran the Streamlit application, indexed the assignment PDF, ran the automated test suite, and checked the evidence returned by the application.

The local test run recorded on 29 October 2026 completed with **64 tests passed**. The uploaded assignment PDF was also extracted and indexed into 15 chunks, with 15 vectors and 15 keyword entries recorded. These are observed verification results; they do not establish that every generated answer is correct.

ChatGPT did not independently verify the full local repository, deploy the application, or confirm the behaviour of an unshared live environment. Code suggestions needed to be applied and tested in the project environment. The developer remains responsible for the submitted code, design decisions, security, reliability, and ability to explain and modify the implementation.

## 5. Review and Validation of AI-Generated Suggestions

AI-generated code and explanations can contain errors, omit project-specific details, or make assumptions about surrounding code. For that reason, development relied on reviewing the available code and running local commands and tests rather than treating each suggestion as automatically correct.

One example was the Single-pass RAG source-display issue. The retrieval function returned wrapper objects containing the actual chunk under a `chunk` key, while the UI and generator expected the chunk fields at the top level. After the UI was adjusted to pass the underlying non-empty chunk dictionaries, the result displayed actual source text and identifiers rather than empty passages.

The project also encountered Gemini API quota/rate-limit responses during the batch answer-evaluation attempt. The incomplete run was not reported as a completed evaluation or used to claim a measured overall answer-accuracy score.

## 6. Limitations and Responsible Use

- AI-assisted code must be reviewed and tested before use.
- LLM-generated answers may be incorrect or unsupported, even when they sound plausible.
- Retrieved source passages and citation identifiers must be checked against the claims they are intended to support.
- An incomplete evaluation run must not be presented as a completed benchmark.
- API keys and other secrets must be kept out of source control.
- The developer remains accountable for the code and must be able to explain the design, retrieval flow, test results, and limitations independently.

## 7. Summary

ChatGPT supported the development process through architecture discussions, code suggestions, debugging, test interpretation, UI improvement, and documentation. Google Gemini powers the application's LLM-based answer generation and multi-step tool-calling workflow, while Sentence Transformers supplies embeddings for semantic retrieval. The developer applied and tested changes locally and remains responsible for the final implementation. The recorded software test suite passed 64 tests, but the broader quantitative evaluation of generated answers remains incomplete because of Gemini API quota/rate-limit issues.
