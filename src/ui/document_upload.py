from pathlib import Path
import tempfile

import streamlit as st

from src.ingestion.file_loader import load_document


PREVIEW_CHAR_LIMIT = 5000


def render_document_upload() -> None:
    """Render the document upload and text-preview section."""

    st.subheader("Add a document")
    st.caption(
        "Upload a TXT, PDF, or DOCX file to extract its text "
        "and inspect the document metadata."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        key="document_upload",
    )

    if st.button(
        "Extract and preview",
        disabled=uploaded_file is None,
        key="extract_document_button",
    ):
        try:
            # Preserve the original filename so the loader records
            # the correct document title and metadata.
            safe_filename = Path(uploaded_file.name).name

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / safe_filename
                temp_path.write_bytes(uploaded_file.getvalue())

                document = load_document(temp_path)

            st.session_state["document_preview"] = document
            st.session_state["document_preview_filename"] = (
                uploaded_file.name
            )
            st.session_state.pop("document_preview_error", None)

        except Exception as exc:
            st.session_state.pop("document_preview", None)
            st.session_state["document_preview_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    error = st.session_state.get("document_preview_error")
    if error:
        st.error(f"Document extraction failed: {error}")

    document = st.session_state.get("document_preview")
    preview_filename = st.session_state.get(
        "document_preview_filename"
    )

    # Avoid showing the previous file's preview when another
    # filename has been selected but not yet extracted.
    if not document or preview_filename != (
        uploaded_file.name if uploaded_file else None
    ):
        return

    metadata = document["metadata"]
    text = document["text"]

    st.success("Document extracted successfully.")
    st.write(f"**Document:** {document['title']}")
    st.code(document["document_id"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", f"{len(text):,}")
    col2.metric(
        "Sentences",
        f"{metadata.get('sentence_count', 0):,}",
    )
    col3.metric(
        "File size",
        f"{metadata.get('file_size_bytes', 0) / 1024:.1f} KB",
    )

    with st.expander("Document metadata"):
        st.json(metadata)

    st.text_area(
        f"Extracted text (first {PREVIEW_CHAR_LIMIT:,} characters)",
        value=text[:PREVIEW_CHAR_LIMIT],
        height=300,
        disabled=True,
    )

    if len(text) > PREVIEW_CHAR_LIMIT:
        st.caption(
            f"Preview truncated. The full extracted text contains "
            f"{len(text):,} characters."
        )