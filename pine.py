import os, time, hashlib
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(".env.pa")

PC_API = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PA_INDEX", "pa")

pc = Pinecone(api_key=PC_API)
index = pc.Index(INDEX_NAME)

def make_id(namespace, title):
    slug = "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
    slug = "-".join([s for s in slug.split("-") if s])[:48].strip("-")
    h = hashlib.sha1(f"{namespace}|{title}|{time.time()}".encode()).hexdigest()[:8]
    return f"{namespace}:{time.strftime('%Y-%m-%d')}:{slug}:{h}"

def upsert(namespace, vectors):
    # vectors: list of {"id": str, "values": [float], "metadata": dict}
    index.upsert(vectors=vectors, namespace=namespace)

def query(namespace, vector, top_k=8, filter_meta=None):
    return index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter_meta or {}
    )

    
