# AI Investigation Engine

An AI-powered investigation system that searches, analyzes, and connects evidence across large volumes of documents to generate structured, source-backed answers.

Unlike a basic document chatbot, the AI Investigation Engine combines semantic search, keyword retrieval, evidence analysis, and tool calling to investigate questions that may require multiple related facts.

## Live Demo and Project

- **Live Application:** https://ai-investigation-engine.streamlit.app/
- **GitHub Repository:** https://github.com/KgothatsoTheko/AI-Investigation-Engine
- **Video Demonstration:** https://www.loom.com/share/04abb120c3e74d42981553c3ef45b5ee

**Deployment verification — 29 October 2026:** The live application successfully displayed 67,507 connected vector chunks and 67,507 keyword records, with hosted PostgreSQL and remote Qdrant configured.

## Overview

The AI Investigation Engine supports evidence-based question answering over a large research corpus and user-uploaded documents. It retrieves relevant information, preserves source references, and is designed to acknowledge when the available evidence is insufficient.

The application provides two investigation modes:

- **Single-pass RAG:** Retrieves relevant evidence using hybrid search and generates an answer grounded in the retrieved passages.
- **Multi-step investigation:** Uses an LLM-driven investigation process with tool calling to perform targeted searches and collect evidence before generating a final answer.

Users can upload documents, inspect extracted text and metadata, index documents for search, investigate questions, and review the evidence supporting the results.

## Key Features

### Document Processing

- Supports TXT, PDF, and DOCX uploads.
- Extracts document text and metadata.
- Splits document content into sentence-grouped chunks for retrieval.
- Stores uploaded document text and metadata in a persistent document library.
- Supports adding, updating, and deleting saved documents.
- Supports indexing, re-indexing, and removing documents from search indexes.

The document library stores extracted text, metadata, and associated document information. Original uploaded file binaries are not currently retained for download.

### Hybrid Search

- Uses Sentence Transformer embeddings for semantic retrieval.
- Uses Qdrant for vector storage and similarity search.
- Uses PostgreSQL full-text search for hosted keyword retrieval.
- Supports SQLite BM25 as the local keyword-search implementation.
- Combines semantic and keyword rankings using Reciprocal Rank Fusion (RRF).
- Searches the main research corpus and uploaded-document indexes.
- Deduplicates repeated evidence and filters empty text chunks.

### Evidence-Based Investigation

- Generates structured answers with supporting findings.
- Displays source passages, document identifiers, and chunk identifiers.
- Supports targeted follow-up searches in multi-step mode.
- Provides an insufficient-evidence response when retrieved evidence does not support an answer.
- Includes handling for LLM errors and API quota or rate limits.

### Evaluation and Testing

- Includes automated unit and integration tests.
- Provides an answer-evaluation workflow.
- Evaluates retrieval quality, answer correctness, and citation quality.
- Includes measurements for different question categories and chunking strategies.
- Documents evaluation limitations and unsuccessful API-dependent experiments.

## Architecture

The system is organized into document ingestion, indexing, retrieval, investigation, generation, and user-interface components.

### Hosted Deployment Architecture

```text
                  Streamlit Community Cloud
                            |
                   AI Investigation Engine
                            |
              +-------------+-------------+
              |                           |
       Document Workflow             Investigation
              |                           |
       Text Extraction             Single-pass / Multi-step
       and Metadata                        |
              |                     Hybrid Retrieval
       Supabase PostgreSQL                 |
       Document Library             +------+------+
              |                     |             |
       Sentence-grouped          Qdrant       Supabase
       Document Chunks             Cloud       PostgreSQL
                                   Semantic      Full-text
                                   Search        Search
                                      |             |
                                      +------+------+
                                             |
                                  Reciprocal Rank Fusion
                                             |
                                      Evidence Chunks
                                             |
                                         Gemini API
                                             |
                                   Structured Answer
                                             |
                                  Findings and Citations
```

### Production Storage

| Component | Production service |
|---|---|
| Main research corpus vectors | Qdrant Cloud |
| Uploaded document vectors | Qdrant Cloud |
| Main corpus keyword index | Supabase PostgreSQL |
| Uploaded document keyword index | Supabase PostgreSQL |
| Saved document text and metadata | Supabase PostgreSQL |

Local development supports SQLite and local Qdrant as fallbacks.

### Main Components

| Component | Responsibility |
|---|---|
| Streamlit | User interface and application hosting |
| Sentence Transformers | Generates document and query embeddings |
| Qdrant Cloud | Stores and searches semantic vectors |
| Supabase PostgreSQL | Persists document data and hosted keyword-search records |
| PostgreSQL Full-Text Search | Provides hosted keyword retrieval |
| SQLite BM25 | Supports local keyword retrieval |
| Reciprocal Rank Fusion | Combines ranked results from different retrieval sources |
| Gemini API | Supports answer generation and multi-step investigation |
| Pytest | Runs automated tests |

The main corpus and uploaded documents use separate search indexes, allowing uploaded content to be managed without replacing the original research corpus.

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **Hosting:** Streamlit Community Cloud
- **Embeddings:** Sentence Transformers
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** Qdrant Cloud
- **Hosted Keyword Retrieval:** PostgreSQL Full-Text Search
- **Local Keyword Retrieval:** SQLite BM25
- **Database:** Supabase PostgreSQL
- **LLM Integration:** Google Gemini API
- **Document Formats:** TXT, PDF, DOCX
- **Testing:** Pytest

## Dataset and Corpus

The main research corpus was built using the [2WikiMultihopQA dataset](https://huggingface.co/datasets/xanhho/2WikiMultihopQA).

**Dataset revision:**

`6ef4eb1bc94ca4a768bb86809fafe777ad795995`

### Corpus Build Statistics

| Measure | Result |
|---|---:|
| Source examples scanned | 10,878 |
| Source passage occurrences | 108,780 |
| Unique text records retained | 50,001 |
| Duplicate occurrences removed | 58,779 |
| Evaluation seed records | 100 |
| Sentence-grouped chunks | 67,507 |
| Fixed-window chunks | 64,744 |

### Hosted Index Verification

| Index | Verified records |
|---|---:|
| Qdrant Cloud — main corpus | 67,507 vectors |
| Supabase PostgreSQL — main corpus | 67,507 keyword records |

These counts were verified following the cloud migrations and through the deployed application's sidebar.

The evaluation dataset includes questions designed to examine single-passage retrieval, multi-hop reasoning, comparison, and inference. Unanswerable and contradictory questions are also part of the broader evaluation objectives, although their full evaluation is not yet complete.

## Getting Started

### Prerequisites

- Python 3.13 or a compatible Python environment.
- Git.
- A Google Gemini API key for LLM-powered investigations.
- Access to local corpus indexes or the configured hosted storage services.

### 1. Clone the Repository

```bash
git clone https://github.com/KgothatsoTheko/AI-Investigation-Engine.git
cd AI-Investigation-Engine
```

### 2. Create a Virtual Environment

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux:**

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

Create a `.env` file in the project root for local development:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
GEMINI_FALLBACK_MODEL=your_configured_fallback_model

QDRANT_URL=your_qdrant_cloud_endpoint
QDRANT_API_KEY=your_qdrant_api_key

SUPABASE_DB_URL=your_postgresql_connection_string
```

The application also supports:

- `GOOGLE_API_KEY` as an alternative to `GEMINI_API_KEY`.
- `DATABASE_URL` as an alternative to `SUPABASE_DB_URL`.

Use the exact model identifiers supported by your Gemini API configuration.

Keep API keys, database passwords, and connection strings out of source control.

For Streamlit Community Cloud, configure the required values in the application's **Settings → Secrets**.

### 5. Local Data and Indexes

The full dataset and generated local indexes are not committed to the repository.

Local development without hosted services requires the relevant data and indexes under:

```text
data/processed/
```

The local fallback indexes include:

```text
data/processed/
├── keyword_index.sqlite
├── qdrant_full/
├── upload_keyword_index.sqlite
├── qdrant_uploads/
└── uploaded_documents.sqlite
```

When hosted services are configured, the application uses Qdrant Cloud and PostgreSQL for the corresponding production storage and retrieval operations. The full local corpus indexes do not need to be bundled into the Streamlit deployment.

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

Open the **Documents** section and upload a TXT, PDF, or DOCX file.

Select **Extract text and preview** to inspect the extracted content and metadata.

When the extracted content looks correct, save the document to the library.

### Step 2: Index the Document

Select a saved document and choose **Index / refresh search index**.

The application generates searchable chunks, embeddings, and keyword-search records.

A saved document is not automatically searchable until it has been indexed.

Re-index an updated document to refresh its searchable content.

### Step 3: Start an Investigation

Open the **Investigate** section, enter a question, and select an investigation mode.

**Single-pass RAG** is suitable for direct questions that can be answered from a focused set of retrieved passages.

**Multi-step investigation** is intended for questions that benefit from several targeted searches, such as questions involving multiple related facts or entities. It can make additional API requests and may consume more Gemini quota.

### Step 4: Review the Result

Review the generated answer, evidence-backed findings, limitations, and retrieved source passages.

Use the displayed chunk and document identifiers to trace claims back to their supporting material.

An answer should not be treated as reliable merely because it is fluent; the retrieved evidence must support its claims.

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

**Latest recorded local verification — 30 October 2026:**

**68 tests passed.**

The test suite covers implemented components such as retrieval, index maintenance, investigation behavior, document storage, uploaded-document indexing, and Qdrant test isolation.

The latest verification was performed with cloud connection environment variables cleared so tests could use isolated local test databases.

## Answer Evaluation

The project includes an evaluation workflow for testing answer generation against prepared evaluation examples.

To evaluate a sample of 40 questions:

```bash
python -m scripts.evaluate_answers --limit 40
```

This workflow requires the relevant local evaluation data and a working Gemini API configuration.

API quotas and rate limits can interrupt evaluation. Incomplete runs should not be interpreted as successful comparisons.

### Recorded Answer-Evaluation Results

| Metric | Result |
|---|---:|
| Exact Match | 0.400 |
| Token F1 | 0.400 |
| Citation Resolution Rate | 1.000 |
| Gold-Fact Citation Precision | 0.925 |
| Gold-Fact Citation Recall | 0.6488 |
| Questions Answered | 18 / 40 |
| Insufficient-Evidence Responses | 22 / 40 |

The results show reliable citation-ID resolution but leave room for improvement in answer correctness and evidence coverage.

The evaluation used a primary Gemini model and a fallback model after provider errors or quota limits. It is therefore not a controlled comparison between the two models.

The saved single-pass versus multi-step experiment also encountered API quota errors and does not provide a valid performance comparison between those modes.

For the complete measurements, methodology, and limitations, see the [Evaluation Report](Evaluation_Report.md).

## Design Decisions

### Hybrid Retrieval

Semantic search can retrieve passages with related meanings even when they use different wording. Keyword search can retrieve exact terms, names, and phrases.

Combining both approaches is intended to improve evidence retrieval.

Reciprocal Rank Fusion combines rankings without requiring semantic and keyword search scores to share the same scale.

### Cloud and Local Storage

The hosted application uses Qdrant Cloud for semantic vector retrieval and Supabase PostgreSQL for keyword search and saved document data.

Local SQLite and Qdrant storage are retained as development and evaluation fallbacks.

Separating production storage from local test databases also prevents automated tests from accidentally modifying hosted indexes.

### Separate Investigation Modes

Single-pass RAG provides a simpler retrieval-and-generation workflow.

Multi-step investigation introduces a planning stage that can call a retrieval tool to search for additional evidence.

Keeping the two modes separate makes their behavior and API usage easier to inspect and evaluate.

### Evidence and Abstention

The system is designed to present source passages alongside generated answers and to avoid unsupported conclusions when the available evidence is insufficient.

Retrieval quality, source coverage, answer correctness, and contradiction handling remain important areas for continued evaluation.

## Limitations

- Answer quality depends on the quality and relevance of retrieved passages.
- The 40-question answer evaluation shows moderate answer matching and incomplete evidence coverage.
- The answer evaluation used a primary model and fallback model, so it is not a controlled model comparison.
- The saved single-pass versus multi-step answer comparison was interrupted by API quota/rate-limit errors.
- Broader evaluation of unanswerable and contradictory questions remains incomplete.
- The document library stores extracted text and metadata; original uploaded file binaries are not currently retained for download.
- Multi-step investigations may require additional Gemini API requests.
- External API quotas, rate limits, and service availability may affect investigations and evaluation runs.

## Reports

- [Evaluation Report](Evaluation_Report.md)
- [AI Usage Report](AI_Usage_Report.md)

---

**AI Investigation Engine** — Retrieve evidence. Connect information. Produce answers that can be investigated.