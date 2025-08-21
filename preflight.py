def preflight():
    import os, importlib, traceback
    ok = True; msgs=[]
    if not os.getenv("PA_INDEX_NAME"): ok=False; msgs.append("PA_INDEX_NAME not set")
    try:
        import pinecone, pine  # your wrapper
        _ = pine.pa_index  # ensure it resolves
    except Exception:
        ok=False; msgs.append("Pinecone index init failed"); traceback.print_exc()
    try:
        import embeddings; _ = embeddings.embed_with("text-embedding-3-large", "ping")
    except Exception:
        msgs.append("Embedder check deferred (will fallback at runtime)")
    return ok, msgs
