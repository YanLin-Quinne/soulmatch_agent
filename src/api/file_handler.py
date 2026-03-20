"""File upload handler - extract text from PDF, TXT, DOCX files."""

import io
from fastapi import UploadFile, HTTPException
from loguru import logger

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_TEXT_LENGTH = 10000
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _get_extension(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""


async def extract_text(file: UploadFile) -> str:
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large ({len(content)} bytes). Max: {MAX_FILE_SIZE} bytes (5MB)")

    try:
        if ext == ".txt":
            text = content.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            text = _extract_pdf(content)
        elif ext == ".docx":
            text = _extract_docx(content)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text extraction failed for {file.filename}: {e}")
        raise HTTPException(422, f"Failed to extract text: {str(e)}")

    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "\n\n[... truncated at 10000 characters]"

    return text


def _extract_pdf(content: bytes) -> str:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text())
    return "\n".join(pages)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
