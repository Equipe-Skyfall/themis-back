import fitz  # pymupdf


def extract_text(pdf_path: str) -> str:
    """Extract and return all text from a PDF file."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()
