import streamlit as st

from src.ui.document_upload import render_document_upload
from src.ui.investigation import render_investigation


st.set_page_config(
    page_title="AI Investigation Engine",
    layout="wide",
)

st.title("AI Investigation Engine")

st.write(
    "Investigate questions by connecting evidence "
    "across multiple documents."
)

with st.sidebar:
    st.header("Search configuration")

    st.write("Main corpus: connected")
    st.write("Uploaded documents: optional search index")

    st.caption(
        "The app searches both the full corpus and indexed "
        "uploaded documents. The embedding model is reused "
        "after its first load."
    )

render_document_upload()

render_investigation()