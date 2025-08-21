# pine.py — PA writes only; PD-Knowledge is read-only
# Safe init (dimension-checked), JSON-safe upserts, and dual-index health queries.

import os, re, uuid, time, json
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------------------------
# Load both env files (won't override already-set vars)
#   .env.pa      → PA index, model, region
#   .env.sciai   → PD-Knowledge index, model
# --------------------------------------------------------------------
load_dotenv(".env.pa")
load_dotenv(".env.sciai")

# PA (private) config
PC_API         = os.getenv("PINECONE_API_KEY")
PA_INDEX_NAME  = os.getenv("PA_INDEX", "pa")
PA_REGION      = os.getenv("PA_REGION", "us-east-1")
PA_EMBED_MODEL = os.getenv("PA_EMBED_MODEL", "text-embedding-3-large")

# PD-Knowledge (external, read-only)
PD_INDEX_NAME  = os.getenv("PD_INDEX", "pd-knowledge")
PD_EMBED_MODEL = os.getenv("PD_EMBED_MODEL", "text-embedding-3-small")  # 1536-d default

# Pinecone client
pc = Pinecone(api_key=PC_API)

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def _model_dim(model: str) -> int:
    m = (model or "").lower()
    if "text-embedding-3-large" in m:  # 3072
        return 3072
    if "text-embedding-3-small" in m:  # 1536
        return 1536
    if "ada-002" in m:                 # 1536
        return 1536
    raise ValueError(f"Unknown embedding model '{model}'. Add its dimension to _model_dim().")

def list_index_names() -> List[str]:
    return [i.name for i in pc.list_indexes()]

def has_index(name: str) -> bool:
    return name in list_index_names()

def describe_index_dim(name: str) -> Optional[int]:
    if not has_index(name):
        return None
    d = pc.describe_index(name)
    try:
        return int(getattr(d, "dimension", 0))
    except Exception:
        return None

def ensure_pa_index(dim: Optional[int] = None, region: Optional[str] = None):
    """Ensure PA index exists with correct dimension based on PA_EMBED_MODEL."""
    target_dim = dim or _model_dim(PA_EMBED_MODEL)
    region = region or PA_REGION
    names = list_index_names()
    if PA_INDEX_NAME in names:
        desc = pc.describe_index(PA_INDEX_NAME)
        current_dim = int(getattr(desc, "dimension", 0))
        if current_dim != target_dim:
            raise RuntimeError(
                f"Index '{PA_INDEX_NAME}' has dim {current_dim}, "
                f"but PA_EMBED_MODEL '{PA_EMBED_MODEL}' expects {target_dim}. "
                f"Delete & recreate the index or change PA_EMBED_MODEL."
            )
    else:
        pc.create_index(
            name=PA_INDEX_NAME,
            dimension=target_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
    return pc.Index(PA_INDEX_NAME)

# bind validated PA index
pa_index = ensure_pa_index()

# --------------------------------------------------------------------
# ID & JSON-safe upsert/query (PA only)
# --------------------------------------------------------------------
def _slug(s: str, max_len=48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").strip().lower()).strip("-")
    return s[:max_len] or "untitled"

def make_id(namespace: str, title: str) -> str:
    """health:2025-08-12:my-title:1a2b3c4d"""
    day = time.strftime("%Y-%m-%d")
    suf = uuid.uuid4().hex[:8]
    return f"{namespace}:{day}:{_slug(title)}:{suf}"

def _jsonable(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata is JSON-serializable (no NaN/None inside lists etc.)."""
    def clean(v):
        if v is None:
            return ""
        if isinstance(v, (str, int, float, bool)):
            # Pinecone allows primitives
            return v
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {str(k): clean(vv) for k, vv in v.items()}
        # fallback to string
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)
    return {str(k): clean(v) for k, v in meta.items()}

def upsert(namespace: str, vectors: List[Dict[str, Any]]):
    """Write into PA only. Each item: {'id','values','metadata'}."""
    # sanitize metadata for each record
    safe = []
    for rec in vectors:
        rid = rec.get("id") or make_id(namespace, "untitled")
        vals = rec.get("values")
        if hasattr(vals, "tolist"):  # e.g., numpy array
            vals = vals.tolist()
        meta = _jsonable(rec.get("metadata") or {})
        safe.append({"id": rid, "values": vals, "metadata": meta})
    pa_index.upsert(vectors=safe, namespace=namespace)

def query(namespace: Optional[str], vector: List[float], top_k=8, filter_meta: Optional[Dict[str, Any]]=None):
    return pa_index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace or None,
        include_metadata=True,
        filter=filter_meta or {}
    )

# --------------------------------------------------------------------
# Cross-index (READ-ONLY) — used for PD-Knowledge queries
# --------------------------------------------------------------------
def query_index(index_name: str, namespace: Optional[str], vector: List[float], top_k=8, filter_meta: Optional[Dict[str, Any]]=None):
    if not has_index(index_name):
        class _R: matches = []
        return _R()
    idx = pc.Index(index_name)
    return idx.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace or None,
        include_metadata=True,
        filter=filter_meta or {}
    )

# --------------------------------------------------------------------
# Dual-index retrieval for Health
#   - Pass *already-embedded* vectors with matching dims:
#       qvec_pa → 3072-d (text-embedding-3-large) for PA
#       qvec_pd → 1536-d (t-e-3-small / ada-002)     for PD-Knowledge
#   - Returns a combined, score-sorted list of dicts with 'origin'
# --------------------------------------------------------------------
def _to_rows(matches, origin: str) -> List[Dict[str, Any]]:
    rows = []
    for m in getattr(matches, "matches", []) or []:
        rows.append({
            "id": m.id,
            "score": float(getattr(m, "score", 0.0) or 0.0),
            "namespace": getattr(m, "namespace", None),
            "origin": origin,  # 'pa' or 'pd-knowledge'
            "metadata": getattr(m, "metadata", {}) or {},
        })
    return rows

def query_health_indexes(qvec_pa: List[float], qvec_pd: Optional[List[float]], top_k: int = 8, pa_namespace: str = "health") -> List[Dict[str, Any]]:
    # 1) PA (private)
    res_pa = query(pa_namespace, qvec_pa, top_k=top_k)
    rows = _to_rows(res_pa, origin="pa")

    # 2) PD-Knowledge (external, read-only)
    if qvec_pd is not None and has_index(PD_INDEX_NAME):
        res_pd = query_index(PD_INDEX_NAME, None, qvec_pd, top_k=top_k)
        rows += _to_rows(res_pd, origin="pd-knowledge")

    # 3) sort & return
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

# --------------------------------------------------------------------
# Diagnostics — optional
# --------------------------------------------------------------------
def dims_summary() -> Dict[str, Any]:
    """Quick dimension report to show in UI if you want."""
    return {
        "PA_INDEX": {
            "name": PA_INDEX_NAME,
            "embed_model": PA_EMBED_MODEL,
            "expected_dim": _model_dim(PA_EMBED_MODEL),
            "actual_dim": describe_index_dim(PA_INDEX_NAME),
            "region": PA_REGION,
        },
        "PD_INDEX": {
            "name": PD_INDEX_NAME,
            "embed_model": PD_EMBED_MODEL,
            "expected_dim": _model_dim(PD_EMBED_MODEL),
            "actual_dim": describe_index_dim(PD_INDEX_NAME),
        }
    }
