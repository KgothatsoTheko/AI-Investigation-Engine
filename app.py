import streamlit as st

from src.ui.document_upload import render_document_upload
from src.ui.investigation import render_investigation


st.set_page_config(
    page_title="AI Investigation Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Investigation Engine")
st.caption(
    "Research questions by finding, connecting, and citing evidence across documents."
)

st.markdown(
    "**Getting started:** add a document, save it, and index it for search. "
    "Then open **Investigate** to ask a question and inspect the evidence "
    "behind the answer."
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
    st.success("Main research corpus is connected.")
    st.info(
        "Uploaded documents become searchable after you choose "
        "**Index / refresh search index** in the Documents tab."
    )
    st.caption(
        "The app searches the main corpus and any uploaded documents "
        "that have been indexed."
    )


documents_tab, investigate_tab = st.tabs(
    ["📚 Documents", "🔎 Investigate"]
)

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
