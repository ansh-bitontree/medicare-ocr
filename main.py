import os
import re
import uuid
import tempfile
from datetime import datetime
from typing import Optional


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes

# ─────────────────────────────────────────────
# Lazy OCR loader
# Server binds port FIRST, model loads after
# ─────────────────────────────────────────────
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            use_textline_orientation=True,
            lang="en",
            device="cpu"
        )
    return _ocr


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="OCR Microservice",
    description="Fallback OCR for scanned PDFs and images. Extracts text and removes PHI.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Poppler: Windows path locally, None on Render (Linux uses system poppler)
POPPLER_PATH = r"poppler\Library\bin"
if not os.path.exists(POPPLER_PATH):
    POPPLER_PATH = None


# ─────────────────────────────────────────────
# PHI De-identification
# ─────────────────────────────────────────────
PHI_PATTERNS = [
    # Patient name after label
    (r"(Patient\s*Name\s*[:\-]\s*)([A-Z][a-z]+,?\s+[A-Z][a-z]+\.?\s*[A-Z]?\.?)",
     r"\1[NAME]"),
    # Standalone name (Lastname, Firstname M.)
    (r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z]\.)?\b",
     "[NAME]"),
    # Email
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
     "[EMAIL]"),
    # Phone — all formats
    (r"\+?1?\s*\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}",
     "[PHONE]"),
]

def deidentify(text: str) -> dict:
    clean = text
    phi_found = {}
    for pattern, replacement in PHI_PATTERNS:
        matches = re.findall(pattern, clean, flags=re.IGNORECASE)
        if matches:
            key = replacement.strip().replace("\\1", "").replace("[", "").replace("]", "").strip()
            phi_found[key] = [
                m if isinstance(m, str) else "".join(m) for m in matches
            ]
        clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
    return {"clean_text": clean, "phi_found": phi_found}


# ─────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────
def run_ocr_on_image(image_path: str) -> list:
    result = get_ocr().predict(image_path)
    lines = []
    for res in result:
        for line in res["rec_texts"]:
            if line.strip():
                lines.append(line.strip())
    return lines


def ocr_pdf(file_bytes: bytes) -> dict:
    kwargs = {"dpi": 200}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    pages = convert_from_bytes(file_bytes, **kwargs)
    pages_result = {}

    for i, page in enumerate(pages):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                page.save(tmp_path, "JPEG")
            lines = run_ocr_on_image(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        raw_text = " ".join(lines)
        deid = deidentify(raw_text)
        pages_result[f"page_{i + 1}"] = {
            "text": deid["clean_text"],
            "phi_found": deid["phi_found"],
            "line_count": len(lines),
        }

    return pages_result


def ocr_image(file_bytes: bytes, filename: str) -> dict:
    ext = os.path.splitext(filename)[-1].lower() or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(file_bytes)
        lines = run_ocr_on_image(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    raw_text = " ".join(lines)
    deid = deidentify(raw_text)
    return {
        "page_1": {
            "text": deid["clean_text"],
            "phi_found": deid["phi_found"],
            "line_count": len(lines),
        }
    }


# ─────────────────────────────────────────────
# Doc type detection
# ─────────────────────────────────────────────
def detect_document_type(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["report", "lab", "blood", "panel", "result"]):
        return "report"
    if any(k in name for k in ["bill", "invoice", "payment", "charge"]):
        return "bill"
    if any(k in name for k in ["prescription", "rx", "medication", "drug"]):
        return "prescription"
    return "unknown"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "ocr-microservice",
        "model": "PP-OCRv5",
        "version": "1.0.0"
    }


@app.post("/ocr/extract")
async def extract(
    file: UploadFile = File(...),
    doc_type: Optional[str] = None
):
    """
    Single file OCR — called by Vercel backend as fallback
    when document is scanned / image-based.
    Accepts: PDF, JPG, JPEG, PNG
    Returns: clean text per page (PHI removed) + metadata
    """
    filename = file.filename or "unknown"
    file_bytes = await file.read()

    if not filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Unsupported file. Allowed: pdf, jpg, jpeg, png")

    resolved_doc_type = doc_type or detect_document_type(filename)

    try:
        pages = ocr_pdf(file_bytes) if filename.lower().endswith(".pdf") else ocr_image(file_bytes, filename)
    except Exception as e:
        raise HTTPException(500, f"OCR processing failed: {str(e)}")

    clean_pages = {k: v["text"] for k, v in pages.items()}
    full_text = " ".join(clean_pages.values())

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "doc_type": resolved_doc_type,
        "total_pages": len(pages),
        "extracted": {
            "pages": clean_pages,
            "full_text": full_text,
        },
        "meta": {
            "ocr_engine": "PP-OCRv5",
            "phi_removed": True,
            "fields_stripped": ["name", "email", "phone"],
            "line_counts": {k: v["line_count"] for k, v in pages.items()}
        }
    }


@app.post("/ocr/batch")
async def batch_extract(files: list[UploadFile] = File(...)):
    """
    Batch OCR — process reports + bills + prescriptions in one call.
    Max 20 files per request.
    """
    if len(files) > 20:
        raise HTTPException(400, "Max 20 files per batch request")

    results = []

    for file in files:
        filename = file.filename or "unknown"
        file_bytes = await file.read()
        doc_type = detect_document_type(filename)

        try:
            pages = ocr_pdf(file_bytes) if filename.lower().endswith(".pdf") else ocr_image(file_bytes, filename)
            clean_pages = {k: v["text"] for k, v in pages.items()}
            results.append({
                "filename": filename,
                "doc_type": doc_type,
                "success": True,
                "total_pages": len(pages),
                "extracted": {
                    "pages": clean_pages,
                    "full_text": " ".join(clean_pages.values()),
                }
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "doc_type": doc_type,
                "success": False,
                "error": str(e)
            })

    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_files": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)