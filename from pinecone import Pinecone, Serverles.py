from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv(".env.pa")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
name = os.getenv("PA_INDEX", "pa")

# Delete old 1024-d index (SAFE if empty)
if name in [i.name for i in pc.list_indexes()]:
    pc.delete_index(name)

# Recreate at 3072-d to match text-embedding-3-large
pc.create_index(
    name=name,
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print("Recreated index", name, "with 3072 dims.")
