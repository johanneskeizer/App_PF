# pinecone_search.py — PA + PD search with auto-fallback on embedding dimension
import os, json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from pinecone import Pinecone

HISTORY_PATH = Path("config/history.json")

# ---------------------------
# Robust env discovery
# ---------------------------
def find_env_file(name: str) -> Path | None:
    """Search for an env file from CWD and this script's dir up to 3 parents."""
    candidates = []
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    for base in {cwd, here} | set(here.parents[:3]) | set(cwd.parents[:3]):
        p = base / name
        if p.exists():
            return p
    return None

# ---------------------------
# History helpers
# ---------------------------
def _load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def _append_history(entries):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = _load_history()
    seen = {e.get('id') for e in hist if 'id' in e}
    new = [e for e in entries if e.get('id') not in seen]
    if not new:
        return
    hist.extend(new)
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f, indent=2)

def fetch_all_entries_from_pinecone():
    return _load_history()

# ---------------------------
# OpenAI client (pull key from .env.pa if present)
# ---------------------------
pa_env = find_env_file(".env.pa")
if pa_env:
    load_dotenv(pa_env)  # OPENAI_API_KEY expected here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(model: str, text: str):
    # returns a list[float] embedding
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

# ---------------------------
# Pinecone helpers
# ---------------------------
def load_pc_env(env_filename: str):
    env_path = find_env_file(env_filename)
    if not env_path:
        raise RuntimeError(f"Env file '{env_filename}' not found from CWD={Path.cwd()} or script dir.")
    values = dotenv_values(env_path)
    api_key = values.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(f"PINECONE_API_KEY missing in {env_path}")

    pc = Pinecone(api_key=api_key)

    # Accept both PINECONE_INDEX and PINECONE_INDEX_NAME
    index_name = values.get("PINECONE_INDEX") or values.get("PINECONE_INDEX_NAME")

    meta = {
        "path": str(env_path),
        "host": values.get("PINECONE_HOST"),
        "index": index_name,
        "index_voice": values.get("PINECONE_INDEX_VOICE"),
        "namespace": values.get("PINECONE_NAMESPACE"),
        # per-env embedding model (optional)
        "embedding_model": values.get("EMBEDDING_MODEL"),
    }
    return pc, meta

def _open_index(pc: Pinecone, name: str | None, host: str | None):
    if not name:
        return None
    try:
        # Avoid describe(); some envs/versions glitch there.
        return pc.Index(name, host=host) if host else pc.Index(name)
    except Exception as e:
        print(f"❌ Unable to open Pinecone index '{name}'"
              f"{' via host ' + host if host else ''}: {e}")
        return None

def _query_index(idx, vector, top_k: int, namespace: str | None, label: str):
    """Query an index; if using a namespace returns no hits, retry default namespace."""
    try:
        res = idx.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
        matches = list(res.matches or [])
        if (not matches) and namespace:
            print(f"ℹ️ {label}: namespace='{namespace}' empty, retrying default namespace…")
            res = idx.query(vector=vector, top_k=top_k, include_metadata=True, namespace=None)
            matches = list(res.matches or [])
        return matches
    except Exception as e:
        # let the caller handle dim mismatches, etc.
        raise

def _query_with_model(idx, query_text: str, primary_model: str, alt_model: str,
                      ns: str | None, top_k: int, label: str):
    """
    Embed with primary model; on Pinecone dim mismatch, retry with alt model.
    Also retries without namespace if the first attempt returns zero hits.
    """
    try:
        vec = embed_text(primary_model, query_text)
        return _query_index(idx, vec, top_k, ns, label=label)
    except Exception as e:
        msg = str(e)
        if "Vector dimension" in msg and "does not match the dimension of the index" in msg:
            print(f"ℹ️ {label}: dim mismatch with {primary_model}; retrying with {alt_model}…")
            try:
                vec = embed_text(alt_model, query_text)
                return _query_index(idx, vec, top_k, ns, label=label)
            except Exception as e2:
                print(f"⚠️ {label}: fallback with {alt_model} failed: {e2}")
                return []
        else:
            print(f"⚠️ {label}: query failed: {e}")
            return []

def _ns_candidates(primary: str | None, aliases_env_name: str):
    # e.g. PINECONE_NAMESPACE_ALIASES_PA=pa_379fc0d529
    aliases = os.getenv(aliases_env_name, "")
    extra = [a.strip() for a in aliases.split(",") if a.strip()]
    seen = set()
    out = []
    for ns in [primary] + extra:
        if ns in seen: 
            continue
        seen.add(ns)
        out.append(ns or None)   # None = default namespace
    return out

# ---------------------------
# Store (defaults to .env.pa)
# ---------------------------
def embed_and_store(text: str, source: str, type_: str):
    pc, meta = load_pc_env(".env.pa")
    print(f"🔧 Using PA env: {meta['path']}")
    index_name = meta.get("index") or "pa"
    host = meta.get("host")
    ns = meta.get("namespace") or None
    model_pa = meta.get("embedding_model") or "text-embedding-3-small"

    print(f"📦 Upsert target -> index='{index_name}', host='{host}', namespace='{ns}', model='{model_pa}'")
    index = _open_index(pc, index_name, host)
    if index is None:
        raise RuntimeError(
            f"No usable Pinecone index for PA. Set PINECONE_INDEX and (for serverless) PINECONE_HOST in {meta['path']}."
        )

    vec = embed_text(model_pa, text)
    uid = f"{type_}-{source}-{uuid4()}"
    index.upsert(
        vectors=[{"id": uid, "values": vec, "metadata": {"text": text, "source": source, "type": type_}}],
        namespace=ns
    )
    print(f"✅ Stored {type_} from {source} -> {uid} into '{index_name}'.")

# ---------------------------
# Search (PA + PD knowledge)
# ---------------------------
def combined_search(query: str, top_k: int = 3):
    all_matches = []

    # --- PA (.env.pa)
    try:
        pc_pa, meta_pa = load_pc_env(".env.pa")
        print(f"🔧 Using PA env: {meta_pa['path']}")
        idx_pa_name = meta_pa.get("index") or "pa"
        idx_pa_host = meta_pa.get("host")
        ns_pa = meta_pa.get("namespace") or None
        model_pa = meta_pa.get("embedding_model") or "text-embedding-3-small"
        alt_pa = "text-embedding-3-large" if model_pa.endswith("small") else "text-embedding-3-small"
        print(f"🔎 PA search -> index='{idx_pa_name}', host='{idx_pa_host}', namespace='{ns_pa}', model='{model_pa}'")
        idx_pa = _open_index(pc_pa, idx_pa_name, idx_pa_host)
        if idx_pa:

            ns_pa_primary = meta_pa.get("namespace") or "pa"
            for ns_try in _ns_candidates(ns_pa_primary, "PINECONE_NAMESPACE_ALIASES_PA"):
             label = f"PA(ns={ns_try or 'default'})"
            m = _query_with_model(idx_pa, query, model_pa, alt_pa, ns_try, top_k, label)
            all_matches.extend(m)

        else:
            print(f"ℹ️ Skipping PA index '{idx_pa_name}'.")
    except Exception as e:
        print(f"⚠️ Could not load .env.pa: {e}")

    # Optional Voice index (same env)
    try:
        idx_voice_name = meta_pa.get("index_voice") if 'meta_pa' in locals() else None
        if idx_voice_name:
            print(f"🔎 Voice search -> index='{idx_voice_name}', host='{idx_pa_host}', namespace='{ns_pa}', model='{model_pa}'")
            idx_voice = _open_index(pc_pa, idx_voice_name, idx_pa_host)
            if idx_voice:
                m = _query_with_model(idx_voice, query, model_pa, alt_pa, ns_pa, top_k, "Voice")
                all_matches.extend(m)
            else:
                print(f"ℹ️ Skipping Voice index '{idx_voice_name}'.")
    except Exception as e:
        print(f"⚠️ Voice index check failed: {e}")

    # --- PD knowledge (.env.sciai)
    try:
        pc_pd, meta_pd = load_pc_env(".env.sciai")
        print(f"🔧 Using PD env: {meta_pd['path']}")
        idx_pd_name = meta_pd.get("index") or "pd-knowledge"
        idx_pd_host = meta_pd.get("host")
        ns_pd = meta_pd.get("namespace") or None
        # default to LARGE for PD; fallback will switch to SMALL if needed
        model_pd = meta_pd.get("embedding_model") or "text-embedding-3-large"
        alt_pd = "text-embedding-3-small" if model_pd.endswith("large") else "text-embedding-3-large"
        print(f"🔎 PD search -> index='{idx_pd_name}', host='{idx_pd_host}', namespace='{ns_pd}', model='{model_pd}'")

        idx_pd = _open_index(pc_pd, idx_pd_name, idx_pd_host)
        if idx_pd:
            m = _query_with_model(idx_pd, query, model_pd, alt_pd, ns_pd, top_k, "PD")
            all_matches.extend(m)
        else:
            print(f"ℹ️ Skipping PD index '{idx_pd_name}'.")
    except Exception as e:
        print(f"❌ Could not load .env.sciai for PD knowledge: {e}")

    return all_matches

# ---------------------------
# An
