# AI Investigation Engine

An AI-powered investigation system that searches, analyzes, and connects evidence across large volumes of documents to generate structured, source-backed answers.

Unlike a basic document chatbot, the AI Investigation Engine combines semantic search, keyword retrieval, evidence analysis, and tool calling to investigate questions that may require multiple pieces of information.

## Overview

The AI Investigation Engine is built to support evidence-based question answering over a large corpus of text records and user-uploaded documents. It aims to retrieve relevant information, preserve source references, and acknowledge when the available evidence is insufficient.

The application provides two investigation modes:

- **Single-pass RAG:** Retrieves relevant evidence using hybrid search and generates an answer grounded in the retrieved passages.
- **Multi-step investigation:** Uses an LLM-driven investigation process with tool calling to perform targeted searches and collect evidence before generating a final answer.

Users can upload documents, inspect extracted text and metadata, index documents for search, investigate questions, and review the evidence supporting the results.

## Demo

https://www.loom.com/share/04abb120c3e74d42981553c3ef45b5ee

## Key Features

### Document Processing
- Supports TXT, PDF, and DOCX uploads.
- Extracts document text and metadata.
- Splits document content into chunks for retrieval.
- Stores uploaded documents in a persistent SQLite-backed document library.
- Supports adding, updating, and deleting saved documents.
- Supports indexing, re-indexing, and removing documents from search indexes.

### Hybrid Search
- Uses sentence-transformer embeddings for semantic retrieval.
- Uses SQLite-backed BM25 keyword search.
- Combines semantic and keyword rankings using reciprocal rank fusion.
- Searches the main corpus and, when available, the uploaded-document indexes.
- Deduplicates repeated evidence and filters empty text chunks.

### Evidence-Based Investigation
- Generates structured answers with supporting findings.
- Displays source passages, document identifiers, and chunk identifiers.
- Supports targeted follow-up searches in multi-step mode.
- Provides an insufficient-evidence response when the retrieved evidence does not support an answer.
- Includes handling for LLM errors and API quota or rate limits.

### Evaluation and Testing
- Includes automated unit and integration tests.
- Provides an answer-evaluation workflow.
- Supports evaluation across question types, including multi-step questions.
- Supports investigation of answer quality, evidence relevance, and limitations.

## Architecture

The system is organized into document ingestion, indexing, retrieval, investigation, generation, and user-interface components.

```text
                   ┌─────────────────────────┐
                   │      Streamlit UI       │
                   └────────────┬────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
          Document Upload               Investigation
                 │                             │
          Text Extraction              Investigation Mode
          and Metadata                  ┌──────┴──────┐
                 │                      │             │
          SQLite Document Store     Single-Pass   Multi-Step
                 │                      │             │
          Document Chunking             └──────┬──────┘
                 │                             │
          Search Indexing               Hybrid Retrieval
                                 ┌─────────────┴────────────┐
                                 │                          │
                          Semantic Search            BM25 Search
                              Qdrant                    SQLite
                                 │                          │
                                 └─────────────┬────────────┘
                                               │
                                     Reciprocal Rank Fusion
                                               │
                                        Evidence Chunks
                                               │
                                        Gemini LLM
                                               │
                                    Structured Answer
                                               │
                                  Findings and Source Passages
```

### Main Components

| Component | Responsibility |
|---|---|
| Streamlit | User interface for document management and investigations |
| Sentence Transformers | Generates embeddings for semantic search |
| Qdrant | Stores and searches document embeddings |
| SQLite and BM25 | Supports keyword-based retrieval |
| Reciprocal Rank Fusion | Combines results from multiple retrieval sources |
| Gemini API | Supports answer generation and multi-step investigation |
| SQLite document store | Persists uploaded documents and their metadata |
| Pytest | Runs automated tests |

The main corpus and uploaded documents use separate search indexes, allowing uploaded content to be managed without replacing the original corpus.

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **Embeddings:** Sentence Transformers
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** Qdrant
- **Keyword retrieval:** SQLite-backed BM25
- **LLM integration:** Google Gemini API
- **Document formats:** TXT, PDF, DOCX
- **Document persistence:** SQLite
- **Testing:** Pytest

## Dataset

The main corpus was built using the [2WikiMultihopQA dataset](https://huggingface.co/datasets/xanhho/2WikiMultihopQA).

The corpus-building workflow was configured to collect at least 50,000 unique text passages. The project also maintains evaluation data for testing retrieval and answer generation.

The intended evaluation coverage includes:

- Questions answerable from a single passage.
- Questions requiring multiple related facts.
- Questions requiring information to be connected across sources.
- Questions whose answers are absent from the available evidence.
- Questions involving contradictory information.

The purpose of these tests is to evaluate not only whether the system produces an answer, but also whether the answer is supported by the evidence retrieved.

## Getting Started

### Prerequisites

- Python 3.13 or a compatible Python environment.
- Git.
- A Google Gemini API key for LLM-powered investigations.
- Sufficient local disk space for the dataset, indexes, and embedding model.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Investigation-Engine
```

Replace the repository URL with your GitHub repository URL.

### 2. Create a Virtual Environment

**Windows**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
GEMINI_FALLBACK_MODEL=your_configured_fallback_model
```

The application also accepts `GOOGLE_API_KEY` as an alternative to `GEMINI_API_KEY`.

Use model identifiers supported by your Gemini API configuration. Keep API keys and other secrets out of source control.

### 5. Verify the Local Data and Indexes

The application expects the processed corpus and its search indexes to be available in:

```text
data/processed/
```

The main retriever expects the following index paths:

```text
data/processed/
├── keyword_index.sqlite
├── qdrant_full/
├── upload_keyword_index.sqlite
└── qdrant_uploads/
```

The uploaded-document index paths are optional. The main corpus indexes must exist before investigations can run.

The full dataset and generated indexes may need to be prepared separately if they are not included in the repository.

### 6. Start the Application

```bash
streamlit run app.py
```

Open the local URL printed in your terminal, normally:

```text
http://localhost:8501
```

## How to Use the Application

### Step 1: Add a Document

Open the document management section and upload a TXT, PDF, or DOCX file.

Extract and preview the document to inspect its text and metadata. Save it to the document library when the extracted content looks correct.

### Step 2: Index the Document

Select the saved document and use the indexing action to generate embeddings and update the keyword and vector indexes.

A saved document is not automatically searchable merely because it exists in the document library. It must be indexed first.

Re-index an updated document to refresh its searchable content.

### Step 3: Start an Investigation

Enter a question in the investigation interface and select an investigation mode.

**Single-pass RAG** is suitable for direct questions that can be answered from a focused set of retrieved passages.

**Multi-step investigation** is intended for questions that benefit from several targeted searches, such as questions involving multiple related facts or entities. It can make additional API requests and may consume more quota.

### Step 4: Review the Result

Review the generated answer, evidence-backed findings, limitations, and retrieved source passages.

Use the displayed chunk and document identifiers to trace claims back to their supporting material. An answer should not be treated as reliable merely because it is fluent; the retrieved evidence must support its claims.

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

The test suite covers implemented components such as retrieval, index maintenance, investigation behavior, and uploaded-document indexing.

**Latest recorded local verification:** 64 tests passed. Re-run the suite after code changes to confirm the current state.

## Answer Evaluation

The project includes an evaluation workflow for testing answer generation against prepared evaluation examples.

For example, to evaluate a sample of 40 questions:

```bash
python -m scripts.evaluate_answers --limit 40
```

This workflow requires the relevant local evaluation data and a working Gemini API configuration. API quotas or rate limits can interrupt evaluation, so incomplete runs should not be interpreted as full evaluation results.

Evaluation results should be reviewed for answer correctness, evidence relevance, citation quality, and appropriate abstention when evidence is insufficient. Record the actual results in the Evaluation Report rather than assuming successful execution proves answer quality.

## Design Decisions

### Hybrid Retrieval

Semantic search can retrieve passages with related meanings even when they use different wording. Keyword search can retrieve exact terms, names, and phrases. Combining both approaches is intended to improve retrieval coverage.

Reciprocal rank fusion combines the rankings without requiring keyword and semantic search scores to share the same scale.

### Local Index Persistence

The project uses local Qdrant storage for vector indexes and SQLite for keyword retrieval and document persistence during local development.

This reduces the infrastructure needed to get started. However, local Qdrant storage has performance limitations for large collections. Production deployment should use an appropriate persistent storage configuration and should be tested at the target corpus size.

### Separate Investigation Modes

Single-pass RAG provides a simpler retrieval-and-generation workflow. Multi-step investigation introduces a planning stage that can call a retrieval tool to search for additional evidence.

Keeping the two modes separate makes their behavior and API usage easier to compare during evaluation.

### Evidence and Abstention

The system is designed to present source passages alongside generated answers and to avoid unsupported conclusions when the available evidence is insufficient.

Retrieval quality, source coverage, answer correctness, and contradiction handling remain important areas to validate through the evaluation framework.

## Limitations

- Answer quality depends on the quality and relevance of retrieved passages.
- Multi-step investigations can require additional Gemini API requests.
- External API quotas, rate limits, and service availability may affect investigations and evaluation runs.
- Local vector storage may become a performance bottleneck as the corpus grows.
- Contradictory information and missing evidence require careful evaluation; these behaviors should not be assumed correct without test results.
- Local execution does not itself satisfy the requirement for a publicly accessible deployment.
- Persistent data on a hosted platform depends on the storage configuration and hosting environment.

## Deployment

**Live Demo:** [Add the public deployment URL here once deployment is available.](https://ai-investigation-engine.streamlit.app/)

---

**AI Investigation Engine** — Retrieve evidence. Connect information. Produce answers that can be investigated.
