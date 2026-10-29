from pathlib import Path
from typing import Any

import streamlit as st

from src.embeddings.encoder import TextEncoder
from src.rag.generator import GeminiAnswerGenerator
from src.rag.investigator import MultiStepInvestigator
from src.rag.retriever import HybridEvidenceRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_encoder() -> TextEncoder:
    """Load the embedding model once and reuse it across reruns."""
    return TextEncoder()


def _to_dict(value: Any) -> dict:
    """Convert a Pydantic model or dictionary to a dictionary."""
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"Expected a dictionary or Pydantic model, got "
        f"{type(value).__name__}."
    )


def _render_investigation_result(result: dict) -> None:
    """Display the answer, findings, search trace, and sources."""

    answer = result["answer"]
    evidence_chunks = result.get("evidence_chunks", [])
    search_log = result.get("search_log", [])

    st.divider()
    st.subheader("Investigation result")

    st.caption(f"Mode: {result['mode']}")

    status = answer.get("status", "unknown")
    st.write(f"**Status:** {status.replace('_', ' ').title()}")

    direct_answer = answer.get("direct_answer", "")
    if direct_answer:
        st.markdown("**Direct answer**")
        st.write(direct_answer)

    answer_text = answer.get("answer", "")
    if answer_text:
        st.markdown("**Explanation**")
        st.markdown(answer_text)

    findings = answer.get("findings", [])

    if findings:
        st.markdown("**Evidence-backed findings**")

        for index, finding in enumerate(findings, start=1):
            finding_data = _to_dict(finding)

            with st.expander(f"Finding {index}"):
                statement = next(
                    (
                        finding_data.get(key)
                        for key in (
                            "claim",
                            "finding",
                            "statement",
                            "description",
                            "text",
                        )
                        if finding_data.get(key)
                    ),
                    None,
                )

                if statement:
                    st.write(statement)
                else:
                    st.json({
                        key: value
                        for key, value in finding_data.items()
                        if key != "source_ids"
                    })

                source_ids = finding_data.get("source_ids", [])

                if source_ids:
                    st.caption("Cited chunk IDs")
                    for source_id in source_ids:
                        st.code(source_id)

    limitations = answer.get("limitations", [])

    if limitations:
        st.markdown("**Limitations**")
        for limitation in limitations:
            st.write(f"- {limitation}")

    if search_log:
        with st.expander("Investigation search trace"):
            st.dataframe(
                search_log,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("**Retrieved source passages**")

    if not evidence_chunks:
        st.info("No source passages were retrieved.")
        return

    for index, chunk in enumerate(evidence_chunks, start=1):
        title = chunk.get("title") or "Untitled source"
        chunk_id = chunk.get("chunk_id", "unknown")
        document_id = chunk.get("document_id", "unknown")

        with st.expander(f"Source {index}: {title}"):
            st.caption(f"Chunk ID: {chunk_id}")
            st.caption(f"Document ID: {document_id}")

            sentence_indices = chunk.get(
                "source_sentence_indices",
                [],
            )

            if sentence_indices:
                st.caption(
                    f"Source sentence indices: {sentence_indices}"
                )

            st.write(chunk.get("text", ""))


def render_investigation() -> None:
    """Render the question form and investigation results."""

    st.subheader("Start an investigation")

    question = st.text_area(
        "What do you want to investigate?",
        placeholder=(
            "Which requirements does the task specify, "
            "and how should the system demonstrate that "
            "it satisfies them?"
        ),
        key="investigation_question",
    )

    mode = st.radio(
        "Investigation mode",
        options=[
            "Single-pass RAG",
            "Multi-step investigation",
        ],
        horizontal=True,
        help=(
            "Single-pass uses one retrieval pass. Multi-step uses "
            "Gemini tool calling to search for additional evidence."
        ),
    )

    st.caption(
        "Multi-step investigation can use more Gemini requests. "
        "Use it when the question requires several linked facts."
    )

    if st.button(
        "Investigate",
        type="primary",
        key="run_investigation",
    ):
        st.session_state.pop("last_investigation", None)

        if not question.strip():
            st.error("Please enter an investigation question.")

        else:
            try:
                with st.spinner(
                    "Searching documents and analyzing evidence..."
                ):
                    # The embedding model loads on the first actual
                    # investigation, not on every page rerun.
                    encoder = get_embedding_encoder()

                    if mode == "Multi-step investigation":
                        investigator = MultiStepInvestigator(
                            data_dir=DATA_DIR,
                            encoder=encoder,
                        )

                        try:
                            investigation = investigator.investigate(
                                question.strip()
                            )
                        finally:
                            investigator.close()

                        answer_data = _to_dict(
                            investigation["answer"]
                        )

                        evidence_chunks = investigation.get(
                            "evidence_chunks",
                            [],
                        )

                        search_log = investigation.get(
                            "search_log",
                            [],
                        )

                    else:
                        retriever = None
                        generator = None

                        try:
                            retriever = HybridEvidenceRetriever(
                                data_dir=DATA_DIR,
                                encoder=encoder,
                            )

                            generator = GeminiAnswerGenerator()

                            retrieved_results = retriever.retrieve(
                                question.strip()
                            )

                            # Unwrap retrieval results so the generator and UI receive
                            # the actual document chunks, not the retrieval wrappers.
                            evidence_chunks = []

                            for result in retrieved_results:
                                chunk = result.get("chunk")

                                if not isinstance(chunk, dict):
                                    continue

                                # Ignore chunks without readable evidence.
                                if not str(chunk.get("text", "")).strip():
                                    continue

                                evidence_chunks.append(chunk)

                            answer = generator.answer(
                                question.strip(),
                                evidence_chunks,
                            )

                            answer_data = _to_dict(answer)
                            search_log = []

                        finally:
                            if retriever is not None:
                                retriever.close()

                            if generator is not None:
                                generator.client.close()

                    st.session_state["last_investigation"] = {
                        "question": question.strip(),
                        "mode": mode,
                        "answer": answer_data,
                        "evidence_chunks": evidence_chunks,
                        "search_log": search_log,
                    }

            except Exception as exc:
                error_text = str(exc)
                upper_error = error_text.upper()

                if any(
                    marker in upper_error
                    for marker in (
                        "RESOURCE_EXHAUSTED",
                        "429",
                        "RATE_LIMIT",
                        "QUOTA",
                    )
                ):
                    st.error(
                        "The Gemini request appears to have hit "
                        "a quota or rate limit. Wait until the "
                        "provider's stated retry time or review "
                        "your model's quota before trying again."
                    )
                else:
                    st.error(
                        "The investigation could not be completed."
                    )

                with st.expander("Technical error details"):
                    st.code(
                        f"{type(exc).__name__}: {error_text}"
                    )

    previous_result = st.session_state.get("last_investigation")

    # Avoid presenting a previous answer as if it addressed a
    # newly edited question or a different investigation mode.
    if (
        previous_result
        and previous_result["question"] == question.strip()
        and previous_result["mode"] == mode
    ):
        _render_investigation_result(previous_result)