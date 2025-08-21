# 1) See which .env.sciai is actually found
from pathlib import Path
from dotenv import dotenv_values
p = Path(__file__).resolve().parent  # or Path.cwd()
candidates = [p/".env.sciai", *p.parents[:3]]
print("Candidates:", [str(x) for x in candidates if (x/".env.sciai").exists()])

# 2) Verify you can open pd-knowledge with that env directly
from pinecone import Pinecone
vals = dotenv_values(".env.sciai")   # adjust if your .env lives elsewhere
pc = Pinecone(api_key=vals["PINECONE_API_KEY"])
idx = pc.Index(vals.get("PINECONE_INDEX","pd-knowledge"), host=vals.get("PINECONE_HOST"))
print(idx.describe_index_stats())
