
import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """
    Produce an initial sentence list from extracted text.

    This is a practical first-pass splitter, not a full linguistic
    sentence parser. Source block boundaries are preserved separately.
    """
    normalized = normalize_whitespace(text)

    if not normalized:
        return []

    return [
        sentence
        for item in re.split(r"(?<=[.!?])\s+", normalized)
        if (sentence := normalize_whitespace(item))
    ]


def _decode_text_file(raw: bytes) -> tuple[str, str]:
    """Decode a text file with explicit encoding fallbacks."""
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "The TXT file encoding could not be identified."
    )


def _extract_pdf_blocks(
    raw: bytes,
) -> tuple[list[str], dict[str, Any]]:
    """Extract text page by page from a PDF."""
    try:
        reader = PdfReader(BytesIO(raw), strict=False)
    except Exception as exc:
        raise ValueError(
            "Could not open the PDF. It may be corrupt or invalid."
        ) from exc

    if reader.is_encrypted:
        raise ValueError(
            "Encrypted PDFs are not supported by this loader."
        )

    blocks = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(
                f"Could not extract text from PDF page {page_number}."
            ) from exc

        blocks.append(text)

    return blocks, {
        "block_type": "page",
        "page_count": len(reader.pages),
    }


def _extract_docx_blocks(raw: bytes) -> list[str]:
    """Extract Word paragraphs and table rows in document order."""
    try:
        document = Document(BytesIO(raw))
    except Exception as exc:
        raise ValueError(
            "Could not open the DOCX file. "
            "It may be corrupt or invalid."
        ) from exc

    blocks = []

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()

            if text:
                blocks.append(text)

        elif isinstance(item, Table):
            # Retain table content because it may contain important
            # factual relationships, names and dates.
            for row in item.rows:
                cells = [
                    normalize_whitespace(cell.text)
                    for cell in row.cells
                ]

                if any(cells):
                    blocks.append(" | ".join(cells))

    return blocks


def _extract_txt_blocks(raw: bytes) -> tuple[list[str], str]:
    """Extract paragraphs from a plain-text file."""
    text, encoding = _decode_text_file(raw)

    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]

    if not blocks and text.strip():
        blocks = [text.strip()]

    return blocks, encoding


def load_document(file_path: Path | str) -> dict[str, Any]:
    """
    Convert one TXT, PDF or DOCX file to a canonical document.

    The document ID is stable for the same filename, allowing a
    future document-management layer to identify reuploads.
    The content hash identifies the particular file contents.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension or '(no extension)'}. "
            "Supported formats: TXT, PDF and DOCX."
        )

    file_size = path.stat().st_size

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            "The file exceeds the 25 MB size limit."
        )

    raw = path.read_bytes()

    content_hash = hashlib.sha256(raw).hexdigest()

    # Re-uploading a file with the same filename identifies the same
    # logical uploaded document. A separate content hash tracks changes.
    normalized_filename = normalize_whitespace(
        path.name
    ).casefold()

    document_id_hash = hashlib.sha256(
        f"uploaded-file:{normalized_filename}".encode("utf-8")
    ).hexdigest()[:24]

    document_id = f"upload_{document_id_hash}"

    if extension == ".txt":
        blocks, encoding = _extract_txt_blocks(raw)
        extraction_metadata = {
            "encoding": encoding,
            "block_type": "paragraph",
        }

    elif extension == ".pdf":
        blocks, extraction_metadata = _extract_pdf_blocks(raw)

    else:
        blocks = _extract_docx_blocks(raw)
        extraction_metadata = {
            "block_type": "paragraph_or_table_row",
        }

    sentences = []
    block_ranges = []

    for block_number, block in enumerate(blocks, start=1):
        block_sentences = split_sentences(block)

        if not block_sentences:
            continue

        start_sentence = len(sentences)
        sentences.extend(block_sentences)

        block_ranges.append(
            {
                "block_number": block_number,
                "start_sentence_index": start_sentence,
                "end_sentence_index": len(sentences) - 1,
            }
        )

    text = normalize_whitespace(" ".join(sentences))

    if not text:
        if extension == ".pdf":
            raise ValueError(
                "No extractable text was found in the PDF. "
                "If it contains scanned pages, OCR will be needed."
            )

        raise ValueError(
            "The document contains no extractable text."
        )

    metadata = {
        "source_type": "uploaded_file",
        "original_filename": path.name,
        "file_format": extension.removeprefix("."),
        "file_size_bytes": file_size,
        "content_sha256": content_hash,
        "sentence_count": len(sentences),
        "block_ranges": block_ranges,
        **extraction_metadata,
    }

    return {
        "document_id": document_id,
        "title": path.stem,
        "text": text,
        "sentences": sentences,
        "metadata": metadata,
        "provenance": [
            {
                "source_type": "uploaded_file",
                "original_filename": path.name,
                "content_sha256": content_hash,
            }
        ],
    }
