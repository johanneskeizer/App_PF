import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from pinecone import Pinecone

# --- add near the top ---
import json, os
from pathlib import Path

HISTORY_PATH = Path("config/history.json")

def _load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def _append_history(entries):
    """Append only NEW entries (by 'id') to history.json to prevent duplicates."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = _load_history()
    seen = {e.get('id') for e in hist if 'id' in e}
    new = [e for e in entries if e.get('id') not in seen]
    if not new:
        return
    hist.extend(new)
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f, indent=2)

def fetch_all_entries_from_pinecone():
    """
    Returns a list of dicts like:
    {"date": "...", "descriptor": "weight", "dimension": "kg", "value": 72, "id": "..."}
    """
    return _load_history()

# Load default (PA) environment
load_dotenv(Path(".env.pa"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"

def embed_text(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding

def load_pc_env(env_path):
    values = dotenv_values(env_path)
    pc = Pinecone(api_key=values["PINECONE_API_KEY"])
    env = values.get("PINECONE_ENV", "us-west-2")  # fallback if not defined
    return pc, env

from uuid import uuid4

def embed_and_store(text, source, type_):
    pc, _ = load_pc_env(".env.pa")
    index = pc.Index("pa-health")
    vector = embed_text(text)
    unique_id = f"{type_}-{source}-{uuid4()}"
    index.upsert(vectors=[{
        "id": unique_id,
        "values": vector,
        "metadata": {
            "text": text,
            "source": source,
            "type": type_,
        }
    }])
    print(f"✅ Embedded and stored {type_} from {source} -> {unique_id}.")

def combined_search(query, top_k=3):
    vector = embed_text(query)

    pc_pa, _ = load_pc_env(".env.pa")
    index_health = pc_pa.Index("pa-health")
    index_voice = pc_pa.Index("pa-voice-notes")

    matches_health = index_health.query(vector=vector, top_k=top_k, include_metadata=True).matches
    matches_voice = index_voice.query(vector=vector, top_k=top_k, include_metadata=True).matches

    try:
        pc_pd, _ = load_pc_env(".env.sciai")
        index_pd = pc_pd.Index("pd-knowledge")
        matches_pd = index_pd.query(vector=vector, top_k=top_k, include_metadata=True).matches
    except Exception as e:
        print(f"⚠️ pd-knowledge index not available: {e}")
        matches_pd = []

    return matches_health + matches_voice + matches_pd

def generate_answer(query, docs):
    context = "\n".join([m.metadata.get('text', '') for m in docs])
    print("🧠 GPT Context Preview:\n", context)
    messages = [
        {"role": "system", "content": (
            "You are a medical assistant that answers based on personal and scientific documents. "
            "If weight and height are mentioned in the context, calculate the BMI as:"
            "\n\nBMI = weight (kg) / (height in meters)^2"
            "\n\nand provide the result along with an interpretation."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        
    )
    return response.choices[0].message.content


