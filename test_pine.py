from pathlib import Path
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv(Path(__file__).with_name(".env.pa"))
k = os.getenv("PINECONE_API_KEY")
print("Key length:", len(k), k[:8] + "...")

pc = Pinecone(api_key=k)
print("Indexes:", [i.name for i in pc.list_indexes()])
