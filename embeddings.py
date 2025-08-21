# embeddings.py (centralize this so the rest of the app imports from here)
def make_embedder():
    try:
        from embed import embed_with as _local_embed  # your local module
        return _local_embed
    except Exception as e:
        # fallback to OpenAI
        from openai import OpenAI
        client = OpenAI()
        def _fallback(model: str, text: str):
            out = client.embeddings.create(model=model, input=[text])
            return [out.data[0].embedding]
        return _fallback

embed_with = make_embedder()
