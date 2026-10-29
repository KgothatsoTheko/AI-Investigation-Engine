from pathlib import Path
import tempfile

import streamlit as st

from src.ingestion.file_loader import load_document
from src.storage.upload_store import SQLiteUploadStore


PREVIEW_CHAR_LIMIT = 5000

store = SQLiteUploadStore()


def render_document_upload() -> None:
    """Render document upload, preview, and persistent library."""

    st.subheader("Add a document")
    st.caption(
        "Upload a TXT, PDF, or DOCX file to extract its text "
        "and inspect the document metadata."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        max_upload_size=25,
        key="document_upload",
    )

    if st.button(
        "Extract and preview",
        disabled=uploaded_file is None,
        key="extract_document_button",
    ):
        try:
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

    # Render the preview only when it belongs to the selected file.
    preview_matches = (
        document is not None
        and uploaded_file is not None
        and preview_filename == uploaded_file.name
    )

    if preview_matches:
        metadata = document["metadata"]
        text = document["text"]

        st.success("Document extracted successfully.")
        st.write(f"**Document:** {document['title']}")
        st.code(document["document_id"])

        existing_document = store.get_document(
            document["document_id"]
        )

        if existing_document:
            st.caption(
                "A document with this ID already exists. "
                "Saving will update its stored content."
            )

        if st.button(
            "Save or update in document library",
            key="save_document_button",
        ):
            try:
                saved_id = store.save_document(document)
                st.session_state["document_save_message"] = (
                    f"Document saved successfully: {saved_id}"
                )
                st.session_state.pop("document_save_error", None)
            except Exception as exc:
                st.session_state["document_save_error"] = str(exc)

        save_message = st.session_state.pop(
            "document_save_message", None
        )
        if save_message:
            st.success(save_message)

        save_error = st.session_state.pop(
            "document_save_error", None
        )
        if save_error:
            st.error(save_error)

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

    # Always render the saved library, even when no file is uploaded.
    st.divider()
    st.subheader("Saved document library")

    saved_documents = store.list_documents()

    if not saved_documents:
        st.info("No documents have been saved yet.")
    else:
        document_labels = {
            item["document_id"]: (
                f'{item["title"]} — '
                f'{item["metadata"].get("original_filename", "uploaded file")}'
            )
            for item in saved_documents
        }

        selected_id = st.selectbox(
            "Choose a saved document",
            options=list(document_labels),
            format_func=lambda document_id: document_labels[document_id],
            key="saved_document_selection",
        )

        selected_document = store.get_document(selected_id)

        if selected_document:
            saved_metadata = selected_document["metadata"]

            st.write(f'**Title:** {selected_document["title"]}')
            st.write(
                f'**Format:** '
                f'{saved_metadata.get("file_format", "unknown")} '
                f'| **Characters:** '
                f'{len(selected_document["text"]):,}'
            )
            st.code(selected_id)

            if st.checkbox(
                "Confirm deletion of this saved document",
                key=f"confirm_delete_{selected_id}",
            ):
                if st.button(
                    "Delete saved document",
                    key=f"delete_document_{selected_id}",
                ):
                    if store.delete_document(selected_id):
                        st.success("Document deleted from the library.")
                        st.rerun()
                    else:
                        st.warning("The document no longer exists.")