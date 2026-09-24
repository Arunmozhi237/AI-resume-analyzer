import fitz  # PyMuPDF
import pdfplumber
from docx import Document
import io

def extract_from_pdf_pymupdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_from_pdf_pdfplumber(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def extract_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()

def extract_text(file_bytes: bytes, filename: str) -> dict:
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith(".pdf"):
            text = extract_from_pdf_pymupdf(file_bytes)
            if not text:
                text = extract_from_pdf_pdfplumber(file_bytes)
            file_type = "pdf"

        elif filename_lower.endswith(".docx"):
            text = extract_from_docx(file_bytes)
            file_type = "docx"

        else:
            return {
                "text": "",
                "file_type": "unsupported",
                "status": "error: unsupported file type. Use PDF or DOCX."
            }

        if not text:
            return {
                "text": "",
                "file_type": file_type,
                "status": "error: no text could be extracted"
            }

        return {
            "text": text,
            "file_type": file_type,
            "status": "success"
        }

    except Exception as e:
        return {
            "text": "",
            "file_type": "unknown",
            "status": f"error: {str(e)}"
        }