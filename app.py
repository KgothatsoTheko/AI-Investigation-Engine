# app.py
import streamlit as st
from src.ui.document_upload import render_document_upload

st.set_page_config(
    page_title="AI Investigation Engine",
    layout="wide",
)

st.title("AI Investigation Engine")

render_document_upload()

st.write(
    "Investigate questions by connecting evidence "
    "across multiple documents."
)

with st.sidebar:
    st.header("Corpus Status")
    st.metric("Indexed records", "Not connected")
    st.caption("Document storage is not configured yet.")

st.subheader("Start an investigation")

question = st.text_area(
    "What do you want to investigate?",
    placeholder=(
        "Which changes contributed to an incident, "
        "which services were affected, and what evidence "
        "supports the conclusion?"
    ),
)

if st.button("Investigate", type="primary"):
    if not question.strip():
        st.error("Please enter an investigation question.")
    else:
        st.info(
            "The interface is working. "
            "The search and investigation engine "
            "will be connected in the next steps."
        )
