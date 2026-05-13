import fitz


def extract_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return len(doc)


def split_pdf(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    """Extract pages from a PDF. Pages are 1-indexed (inclusive)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        dst = fitz.open()
        dst.insert_pdf(src, from_page=start_page - 1, to_page=end_page - 1)
        result = dst.tobytes()
        dst.close()
        return result
