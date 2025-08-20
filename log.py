# log.py
import os, csv, json, time, hashlib
from pathlib import Path

LOG_DIR = Path("logs")
CSV_PATH = LOG_DIR / "ingest_log.csv"
JSONL_PATH = LOG_DIR / "ingest_log.jsonl"
INDEX_PATH = LOG_DIR / "ingest_index.txt"  # content_hash registry

LOG_DIR.mkdir(exist_ok=True, parents=True)

def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def _ensure_csv_header():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_utc","namespace","type","title","vector_id",
                "content_hash","chars","model","status","error",
                "source","filename"
            ])

def load_hash_index() -> set:
    if not INDEX_PATH.exists():
        return set()
    return set(h.strip() for h in INDEX_PATH.read_text(encoding="utf-8").splitlines() if h.strip())

def add_hash_to_index(h: str):
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(h + "\n")

def log_ingest(namespace, doc_type, title, vector_id, text, model,
               status="ok", error="", source="manual", filename=""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    h = content_hash(text or "")
    _ensure_csv_header()

    # CSV
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([ts, namespace, doc_type, title, vector_id, h, len(text or ""),
                    model, status, error, source, filename])

    # JSONL
    rec = {
        "ts_utc": ts, "namespace": namespace, "type": doc_type, "title": title,
        "vector_id": vector_id, "content_hash": h, "chars": len(text or ""),
        "model": model, "status": status, "error": error,
        "source": source, "filename": filename,
    }
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if status == "ok":
        add_hash_to_index(h)

def is_duplicate(text: str) -> bool:
    return content_hash(text or "") in load_hash_index()
