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
        f"Expected a dictionary or Pydantic model, got {type(value).__name__}."
    )


def _normalize_evidence_chunks(items: list[dict]) -> list[dict]:
    """Accept either retrieval wrappers or plain evidence chunks."""
    normalized = []

    for item in items:
        if not isinstance(item, dict):
            continue

        nested_chunk = item.get("chunk")
        chunk = nested_chunk if isinstance(nested_chunk, dict) else item

        if str(chunk.get("text", "")).strip():
            normalized.append(chunk)

    return normalized


def _render_investigation_result(result: dict) -> None:
    """Display the answer, findings, search trace, and source passages."""
    answer = result["answer"]
    evidence_chunks = result.get("evidence_chunks", [])
    search_log = result.get("search_log", [])

    st.divider()
    st.subheader("Investigation result")
    st.caption(f"Investigation mode: {result['mode']}")

    status = str(answer.get("status", "unknown"))
    status_normalized = status.casefold().replace("_", " ")

    if status_normalized == "answered":
        st.success("Answer status: Answered")
    elif "insufficient" in status_normalized:
        st.warning("Answer status: Insufficient evidence")
    else:
        st.info(f"Answer status: {status.replace('_', ' ').title()}")

    with st.container(border=True):
        st.markdown("### Answer")
        direct_answer = answer.get("direct_answer", "")
        answer_text = answer.get("answer", "")

        if direct_answer:
            st.markdown("**Direct answer**")
            st.write(direct_answer)

        if answer_text:
            st.markdown("**Explanation**")
            st.markdown(answer_text)

        if not direct_answer and not answer_text:
            st.info("No answer text was returned. Check the technical error or evidence below.")

    findings = answer.get("findings", [])
    if findings:
        st.markdown("### Evidence-backed findings")
        st.caption("Expand a finding to inspect its claim and cited chunk IDs.")

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
                    st.caption("Cited source chunk IDs")
                    st.code("\n".join(str(source_id) for source_id in source_ids))

    limitations = answer.get("limitations", [])
    if limitations:
        with st.expander("Limitations and missing evidence"):
            for limitation in limitations:
                st.write(f"- {limitation}")

    if search_log:
        with st.expander("How the engine searched", expanded=False):
            st.caption(
                "For multi-step investigations, this shows the queries the planner ran "
                "and how many new evidence chunks each search found."
            )
            st.dataframe(search_log, width="stretch", hide_index=True)

    st.markdown("### Retrieved source passages")
    st.caption(
        "These are the passages supplied to the answer generator. Expand a source "
        "to inspect its text and provenance."
    )

    if not evidence_chunks:
        st.warning("No source passages were retrieved for this investigation.")
        return

    for index, chunk in enumerate(evidence_chunks, start=1):
        title = chunk.get("title") or "Untitled source"
        chunk_id = chunk.get("chunk_id", "unknown")
        document_id = chunk.get("document_id", "unknown")
        text = str(chunk.get("text", "")).strip()

        with st.expander(f"Source {index}: {title}"):
            col1, col2 = st.columns(2)
            col1.caption("Chunk ID")
            col1.code(str(chunk_id))
            col2.caption("Document ID")
            col2.code(str(document_id))

            sentence_indices = chunk.get("source_sentence_indices", [])
            if sentence_indices:
                st.caption(f"Source sentence indices: {sentence_indices}")

            if text:
                st.write(text)
            else:
                st.warning("This source has no readable text.")


def render_investigation() -> None:
    """Render the question form and investigation results."""
    st.caption(
        "Tip: use Single-pass RAG for a straightforward question. "
        "Use Multi-step investigation when you need to connect several facts "
        "or search for missing pieces of evidence."
    )

    with st.form("investigation_form"):
        question = st.text_area(
            "Your investigation question",
            placeholder=(
                "Example: What are the mandatory project requirements, "
                "and which deliverables must be submitted?"
            ),
            height=110,
            key="investigation_question",
        )

        mode = st.radio(
            "Investigation approach",
            options=[
                "Single-pass RAG",
                "Multi-step investigation",
            ],
            horizontal=True,
            help=(
                "Single-pass retrieves evidence once. Multi-step uses a planner "
                "and tool calling to run focused follow-up searches."
            ),
        )

        submitted = st.form_submit_button(
            "Run investigation",
            type="primary",
            use_container_width=False,
        )

    if mode == "Single-pass RAG":
        st.caption("One retrieval pass followed by answer generation.")
    else:
        st.caption(
            "The planner may run several targeted searches. This can use more Gemini requests."
        )

    if submitted:
        st.session_state.pop("last_investigation", None)

        if not question.strip():
            st.error("Enter a question before starting the investigation.")
        else:
            try:
                with st.spinner("Searching documents and analyzing evidence..."):
                    encoder = get_embedding_encoder()

                    if mode == "Multi-step investigation":
                        investigator = MultiStepInvestigator(
                            data_dir=DATA_DIR,
                            encoder=encoder,
                        )
                        try:
                            investigation = investigator.investigate(question.strip())
                        finally:
                            investigator.close()

                        answer_data = _to_dict(investigation["answer"])
                        evidence_chunks = _normalize_evidence_chunks(
                            investigation.get("evidence_chunks", [])
                        )
                        search_log = investigation.get("search_log", [])

                    else:
                        retriever = None
                        generator = None
                        try:
                            retriever = HybridEvidenceRetriever(
                                data_dir=DATA_DIR,
                                encoder=encoder,
                            )
                            generator = GeminiAnswerGenerator()

                            retrieved_results = retriever.retrieve(question.strip())
                            evidence_chunks = _normalize_evidence_chunks(
                                retrieved_results
                            )

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
                        "The Gemini request appears to have hit a quota or rate limit. "
                        "Check the provider's retry time or model quota before trying again."
                    )
                else:
                    st.error("The investigation could not be completed.")

                with st.expander("Technical error details"):
                    st.code(f"{type(exc).__name__}: {error_text}")

    previous_result = st.session_state.get("last_investigation")
    if (
        previous_result
        and previous_result["question"] == question.strip()
        and previous_result["mode"] == mode
    ):
        _render_investigation_result(previous_result)
