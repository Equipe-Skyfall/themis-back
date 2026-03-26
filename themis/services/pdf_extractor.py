import fitz  # pymupdf
import tempfile
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """Extract and return all text from a PDF file path."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Write PDF bytes to a temp file, extract text, then clean up."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return extract_text(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
