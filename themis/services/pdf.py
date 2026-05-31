import tempfile

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


# ── Path-based variants (lower memory, for large PDFs) ───────────────────────


def page_count_path(path: str) -> int:
    import pikepdf

    pdf = pikepdf.Pdf.open(path)
    count = len(pdf.pages)
    pdf.close()
    return count


def extract_text_path(path: str) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def split_pdf_to_path(src_path: str, start_page: int, end_page: int) -> str:
    """Extract pages to a new temp file using pikepdf (no image decompression).

    Caller is responsible for deleting the returned file.
    """
    import pikepdf

    pdf = pikepdf.Pdf.open(src_path)
    dst = pikepdf.Pdf.new()
    dst.pages.extend(pdf.pages[start_page - 1:end_page])
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    dst.save(tmp.name)
    dst.close()
    pdf.close()
    tmp.close()
    return tmp.name
