# ingest_utils.py
from __future__ import annotations
import io, csv, json, tempfile
from typing import List, Tuple
from pathlib import Path

from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
from pdf2image import convert_from_bytes
import pytesseract

def read_txt(b: bytes) -> str:
    return b.decode("utf-8", errors="ignore").strip()

def read_pdf(b: bytes) -> str:
    # try native text first
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages).strip()
    if text:
        return text
    # OCR fallback
    images = convert_from_bytes(b)  # needs poppler
    ocr_pages = []
    for img in images:
        ocr_pages.append(pytesseract.image_to_string(img))
    return "\n\n".join(ocr_pages).strip()

def read_rtf(b: bytes) -> str:
    return rtf_to_text(b.decode("utf-8", errors="ignore")).strip()

def read_csv(b: bytes, per_row: bool = True) -> List[Tuple[str, str]]:
    """
    Return list of (title, text). If per_row=False, returns one aggregated doc.
    """
    s = b.decode("utf-8", errors="ignore")
    rdr = csv.reader(io.StringIO(s))
    rows = list(rdr)
    if not rows:
        return []
    header = rows[0]
    docs: List[Tuple[str, str]] = []
    for i, row in enumerate(rows[1:], start=1):
        pairs = []
        for h, v in zip(header, row):
            pairs.append(f"{h}: {v}")
        body = "\n".join(pairs).strip()
        if per_row:
            docs.append((f"row-{i}", body))
        else:
            docs.append((f"rows-1..{len(rows)-1}", body))
            break
    return docs

def read_file(name: str, data: bytes) -> List[Tuple[str, str]]:
    """
    Return list of (title, text) to ingest.
    """
    lname = name.lower()
    if lname.endswith(".txt"):
        return [(Path(name).stem, read_txt(data))]
    if lname.endswith(".pdf"):
        return [(Path(name).stem, read_pdf(data))]
    if lname.endswith(".rtf"):
        return [(Path(name).stem, read_rtf(data))]
    if lname.endswith(".csv"):
        return read_csv(data, per_row=True)
    # default: treat as text
    return [(Path(name).stem, read_txt(data))]
