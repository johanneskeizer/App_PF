import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(".env.pa")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

def embed(texts):
    if isinstance(texts, str): texts = [texts]
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]
