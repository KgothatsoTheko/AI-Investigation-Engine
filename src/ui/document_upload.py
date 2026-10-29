from pathlib import Path
import tempfile

import streamlit as st

from src.ingestion.file_loader import load_document
from src.ingestion.upload_indexer import UploadIndexer
from src.storage.upload_store import SQLiteUploadStore


PREVIEW_CHAR_LIMIT = 5000
store = SQLiteUploadStore()


def render_document_upload() -> None:
    """Render document upload, library, indexing, and document management."""

    st.subheader("Step 1 — Upload and preview")
    st.caption(
        "Supported formats: PDF, DOCX, and TXT. Files can be up to 25 MB. "
        "Extract the text first so you can review it before saving."
    )

    uploaded_file = st.file_uploader(
        "Select a document",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        max_upload_size=25,
        key="document_upload",
    )

    if st.button(
        "Extract text and preview",
        type="primary",
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
            st.session_state["document_preview_filename"] = uploaded_file.name
            st.session_state.pop("document_preview_error", None)
            st.session_state.pop("document_preview_error_filename", None)
            st.session_state.pop("document_save_error", None)
            st.session_state.pop("document_save_message", None)

        except Exception as exc:
            st.session_state.pop("document_preview", None)
            st.session_state["document_preview_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            st.session_state["document_preview_error_filename"] = (
                uploaded_file.name if uploaded_file else None
            )

    current_filename = uploaded_file.name if uploaded_file else None
    error = st.session_state.get("document_preview_error")
    error_filename = st.session_state.get("document_preview_error_filename")

    if error and error_filename == current_filename:
        st.error(f"Document extraction failed: {error}")

    document = st.session_state.get("document_preview")
    preview_filename = st.session_state.get("document_preview_filename")
    preview_matches = (
        document is not None
        and uploaded_file is not None
        and preview_filename == uploaded_file.name
    )

    if preview_matches:
        metadata = document.get("metadata", {})
        text = document.get("text", "")

        st.success("Text extracted successfully. Review it, then save the document.")
        st.markdown(f"**Document title:** {document.get('title', 'Untitled document')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Characters extracted", f"{len(text):,}")
        col2.metric("Sentences detected", f"{metadata.get('sentence_count', 0):,}")
        col3.metric(
            "File size",
            f"{metadata.get('file_size_bytes', 0) / 1024:.1f} KB",
        )

        with st.expander("Preview extracted text"):
            st.text_area(
                f"First {PREVIEW_CHAR_LIMIT:,} characters",
                value=text[:PREVIEW_CHAR_LIMIT],
                height=260,
                disabled=True,
                key="document_text_preview",
            )
            if len(text) > PREVIEW_CHAR_LIMIT:
                st.caption(
                    f"Preview shortened for readability. The full document "
                    f"contains {len(text):,} extracted characters."
                )

        with st.expander("View extracted metadata"):
            st.json(metadata)

        existing_document = store.get_document(document["document_id"])
        if existing_document:
            st.caption(
                "A saved document with this ID already exists. Saving will "
                "update its stored content."
            )

        if st.button(
            "Save document to library",
            type="primary",
            key="save_document_button",
        ):
            try:
                saved_id = store.save_document(document)
                st.session_state["document_save_message"] = (
                    f"Document saved. ID: {saved_id}"
                )
                st.session_state.pop("document_save_error", None)
            except Exception as exc:
                st.session_state["document_save_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                st.session_state.pop("document_save_message", None)

    save_message = st.session_state.get("document_save_message")
    if save_message and preview_matches:
        st.success(save_message)

    save_error = st.session_state.get("document_save_error")
    if save_error and preview_matches:
        st.error(f"Could not save document: {save_error}")

    st.divider()
    st.subheader("Step 2 — Saved document library")
    st.caption(
        "Choose a saved document to review its details and manage its search index. "
        "Saving a document does not automatically make it searchable."
    )

    delete_message = st.session_state.pop("document_delete_message", None)
    if delete_message:
        st.success(delete_message)

    saved_documents = store.list_documents()

    if not saved_documents:
        st.info(
            "Your library is empty. Upload a document above, extract its text, "
            "and select **Save document to library**."
        )
        return

    document_labels = {
        item["document_id"]: (
            f"{item['title']} — "
            f"{item['metadata'].get('original_filename', 'uploaded file')}"
        )
        for item in saved_documents
    }

    selected_id = st.selectbox(
        "Select a saved document",
        options=list(document_labels),
        format_func=lambda document_id: document_labels[document_id],
        key="saved_document_selection",
    )

    selected_document = store.get_document(selected_id)
    if not selected_document:
        st.warning("That document could not be loaded from the library. Refresh the page and try again.")
        return

    saved_metadata = selected_document.get("metadata", {})
    selected_text = selected_document.get("text", "")

    with st.container(border=True):
        st.markdown(f"### {selected_document.get('title', 'Untitled document')}")
        st.caption(saved_metadata.get("original_filename", "Uploaded file"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Format", str(saved_metadata.get("file_format", "unknown")).upper())
        col2.metric("Characters", f"{len(selected_text):,}")
        col3.metric(
            "Saved status",
            "Saved",
        )

        with st.expander("View saved document text"):
            st.text_area(
                "Saved text preview",
                value=selected_text[:PREVIEW_CHAR_LIMIT],
                height=220,
                disabled=True,
                key=f"saved_text_preview_{selected_id}",
            )
            if len(selected_text) > PREVIEW_CHAR_LIMIT:
                st.caption(
                    f"Showing {PREVIEW_CHAR_LIMIT:,} of {len(selected_text):,} characters."
                )

    st.subheader("Step 3 — Make this document searchable")
    st.write(
        "Indexing creates searchable chunks for both keyword search and semantic search. "
        "Re-index after updating a saved document."
    )

    if st.button(
        "Index / refresh search index",
        type="primary",
        key=f"index_upload_{selected_id}",
    ):
        indexer = None
        st.session_state.pop("last_upload_index_result", None)
        st.session_state.pop("last_upload_index_error", None)

        try:
            with st.spinner("Generating embeddings and updating search indexes..."):
                indexer = UploadIndexer(document_store=store)
                result = indexer.index_document(selected_id)

            st.session_state["last_upload_index_result"] = {
                "document_id": selected_id,
                "result": result,
            }
        except Exception as exc:
            st.session_state["last_upload_index_error"] = {
                "document_id": selected_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if indexer is not None:
                indexer.close()

    last_index_result = st.session_state.get("last_upload_index_result")
    if last_index_result and last_index_result.get("document_id") == selected_id:
        result = last_index_result["result"]
        st.success(
            f"Search index ready: {result.get('chunks_indexed', 0):,} chunks indexed."
        )
        with st.expander("View indexing details"):
            st.json(result)

    last_index_error = st.session_state.get("last_upload_index_error")
    if last_index_error and last_index_error.get("document_id") == selected_id:
        st.error(f"Indexing failed: {last_index_error['message']}")

    with st.expander("Advanced: remove this document from search"):
        st.caption(
            "This removes the document's indexed chunks, but keeps the saved original in your library."
        )
        confirm_unindex = st.checkbox(
            "I understand this document will no longer be searchable until it is indexed again.",
            key=f"confirm_unindex_{selected_id}",
        )

        if st.button(
            "Remove from search indexes",
            key=f"unindex_upload_{selected_id}",
            disabled=not confirm_unindex,
        ):
            indexer = None
            try:
                with st.spinner("Removing indexed chunks..."):
                    indexer = UploadIndexer(document_store=store)
                    result = indexer.delete_indexed_document(selected_id)
                st.session_state["last_upload_index_result"] = None
                st.session_state["last_upload_index_error"] = None
                st.success("Removed this document from the search indexes.")
                with st.expander("Removal details"):
                    st.json(result)
            except Exception as exc:
                st.error(f"Removal failed: {type(exc).__name__}: {exc}")
            finally:
                if indexer is not None:
                    indexer.close()

    with st.expander("Danger zone: permanently delete this saved document"):
        st.warning(
            "This deletes the saved original and removes its indexed chunks. "
            "You will need to upload it again to use it."
        )
        confirm_delete = st.checkbox(
            "I understand this deletion cannot be undone from the app.",
            key=f"confirm_delete_{selected_id}",
        )

        if st.button(
            "Delete saved document",
            key=f"delete_document_{selected_id}",
            disabled=not confirm_delete,
        ):
            indexer = None
            try:
                with st.spinner("Deleting the document and its search index..."):
                    indexer = UploadIndexer(document_store=store)
                    result = indexer.delete_saved_document(selected_id)

                st.session_state["document_delete_message"] = (
                    f"Deleted '{result.get('title', selected_document.get('title', selected_id))}'. "
                    f"Removed {result.get('keyword_chunks_removed', 0)} keyword chunks "
                    f"and {result.get('vectors_removed', 0)} vectors."
                )
                st.session_state.pop("last_upload_index_result", None)
                st.session_state.pop("last_upload_index_error", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Deletion failed: {type(exc).__name__}: {exc}")
            finally:
                if indexer is not None:
                    indexer.close()
