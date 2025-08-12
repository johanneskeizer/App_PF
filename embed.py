import os
from openai import OpenAI
from dotenv import load_dotenv

# Load PA env by default; PA_App will also pass explicit models when needed
load_dotenv(".env.pa")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default for PA (3072-d)
PA_EMBED_MODEL = os.getenv("PA_EMBED_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-large"))

def embed(texts):
    """Embed with default PA model."""
    return embed_with(PA_EMBED_MODEL, texts)

def embed_with(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    res = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]
