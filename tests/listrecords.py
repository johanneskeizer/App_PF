from dotenv import dotenv_values
from pinecone import Pinecone

v = dotenv_values(".env.pa")
pc = Pinecone(api_key=v["PINECONE_API_KEY"])
index_name = v.get("PINECONE_INDEX","pa")
host = v.get("PINECONE_HOST")
idx = pc.Index(index_name, host=host) if host else pc.Index(index_name)

desc = getattr(idx, "describe_index_stats", None)
if callable(desc):
    stats = desc()
    ns = stats.get("namespaces", {})
    print("Counts by namespace:", {k: v.get("vectorCount") for k, v in ns.items()})
else:
    print("describe_index_stats unavailable in this client version")