# pd_diag.py
import sys
from pathlib import Path
from dotenv import dotenv_values
from pinecone import Pinecone

env_path = Path(".env.sciai")
if not env_path.exists():
    # try parent dirs up to 3 levels (common when starting app from subdir)
    found = None
    base = Path.cwd()
    for p in [base, *base.parents[:3]]:
        cand = p / ".env.sciai"
        if cand.exists():
            found = cand
            break
    if not found:
        print(f"❌ .env.sciai not found from {Path.cwd()}")
        sys.exit(1)
    env_path = found

vals = dotenv_values(env_path)
print(f"✅ Using .env.sciai at: {env_path}")
print("   keys:", [k for k in vals.keys() if k.startswith("PINECONE")])

api_key = vals.get("PINECONE_API_KEY")
index_name = vals.get("PINECONE_INDEX", "pd-knowledge")
host = vals.get("PINECONE_HOST")
ns   = vals.get("PINECONE_NAMESPACE")

if not api_key:
    print("❌ PINECONE_API_KEY missing in .env.sciai")
    sys.exit(1)

pc = Pinecone(api_key=api_key)

# open by host if provided (works cross-project)
try:
    idx = pc.Index(index_name, host=host) if host else pc.Index(index_name)
    stats = idx.describe_index_stats()
    print(f"✅ Opened index '{index_name}'{' via host ' + host if host else ''}")
    print("🧾 Stats (top-level keys):", list(stats.keys()))
    ns_map = (stats.get("namespaces") or {})
    print("🗂  Namespaces & counts:", {k: v.get("vectorCount") for k, v in ns_map.items()})
    if ns and ns not in ns_map:
        print(f"⚠️ Namespace '{ns}' not present in this index. Consider clearing PINECONE_NAMESPACE in .env.sciai.")
except Exception as e:
    print(f"❌ Could not open '{index_name}'"
          f"{' via host ' + str(host) if host else ''}: {e}")
    print("   Fix: ensure API key belongs to the project that owns the index, and set PINECONE_HOST for serverless.")
