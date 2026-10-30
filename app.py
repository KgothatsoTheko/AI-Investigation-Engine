import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Investigation Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load local .env settings, then overlay any values configured in Streamlit Cloud Secrets.
load_dotenv()
SECRET_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "SUPABASE_DB_URL",
    "DATABASE_URL",
)
for secret_name in SECRET_KEYS:
    try:
        if secret_name in st.secrets and not os.getenv(secret_name):
            os.environ[secret_name] = str(st.secrets[secret_name])
    except Exception:
        # Local runs may not have a Streamlit secrets file.
        pass

from src.retrieval.postgres_keyword import PostgresKeywordIndex
from src.retrieval.sqlite_bm25 import SQLiteBM25Index
from src.storage.vector_store import QdrantVectorStore
from src.ui.document_upload import render_document_upload
from src.ui.investigation import render_investigation

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_main_corpus_status() -> tuple[int, int, str | None]:
    """Return vector count, keyword count, and a safe error label if any."""
    vector_count = 0
    keyword_count = 0
    vector_store = None
    keyword_index = None
    database_url = (
        os.getenv("SUPABASE_DB_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    remote_qdrant = bool(os.getenv("QDRANT_URL", "").strip())

    try:
        vector_path = DATA_DIR / "qdrant_full"
        if remote_qdrant or vector_path.exists():
            vector_store = QdrantVectorStore(
                database_path=vector_path,
                collection_name="investigation_chunks",
            )
            if vector_store.has_collection():
                vector_count = vector_store.count_points()
    except Exception:
        return vector_count, keyword_count, "Vector database unavailable"
    finally:
        if vector_store is not None:
            vector_store.close()

    try:
        if database_url:
            keyword_index = PostgresKeywordIndex(
                database_url=database_url,
                source_type="main",
            )
            keyword_count = keyword_index.count_documents()
        else:
            keyword_path = DATA_DIR / "keyword_index.sqlite"
            if keyword_path.exists():
                keyword_index = SQLiteBM25Index(keyword_path)
                keyword_count = keyword_index.count_documents()
    except Exception:
        return vector_count, keyword_count, "Keyword database unavailable"
    finally:
        if keyword_index is not None:
            keyword_index.close()

    return vector_count, keyword_count, None


st.title("AI Investigation Engine")
st.caption(
    "Research questions by finding, connecting, and citing evidence across documents."
)
st.markdown(
    "**Getting started:** add a document, save it, and index it for search. "
    "Then open **Investigate** to ask a question and inspect the evidence behind the answer."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "**1 · Documents**\n\n"
        "Upload a PDF, DOCX, or TXT file. Preview it, save it, then index it.\n\n"
        "**2 · Investigate**\n\n"
        "Ask a question. Review the answer, cited findings, and retrieved passages."
    )
    st.divider()
    st.subheader("Search sources")
    vector_count, keyword_count, status_error = get_main_corpus_status()
    if status_error:
        st.warning(f"Main corpus status: {status_error}.")
    elif vector_count and keyword_count:
        st.success(
            f"Main corpus connected: {vector_count:,} vector chunks and "
            f"{keyword_count:,} keyword records."
        )
    elif vector_count:
        st.warning(
            f"Vector corpus connected ({vector_count:,} chunks), but the main keyword index "
            "is not populated. Main-corpus search will be semantic-only until keyword data is migrated."
        )
    else:
        st.warning(
            "Main research corpus is not connected. You can still upload and index documents below."
        )

    hosted_database = bool(
        os.getenv("SUPABASE_DB_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if hosted_database:
        st.success("Hosted PostgreSQL storage is configured.")
    else:
        st.warning(
            "Uploaded-document storage is local. On Streamlit Community Cloud, "
            "local files may be lost after a restart; configure SUPABASE_DB_URL for persistence."
        )
    if os.getenv("QDRANT_URL", "").strip():
        st.caption("Vector storage: configured remote Qdrant endpoint.")
    else:
        st.caption("Vector storage: local Qdrant mode.")

documents_tab, investigate_tab = st.tabs(["Documents", "Investigate"])
with documents_tab:
    st.header("1. Prepare your documents")
    st.write(
        "Upload and preview source material, save documents to your library, "
        "and index them so the investigation can search their contents."
    )
    render_document_upload()
with investigate_tab:
    st.header("2. Ask a question")
    st.write(
        "Ask a focused question. The engine will retrieve relevant passages "
        "and produce an answer grounded in that evidence."
    )
    render_investigation()
