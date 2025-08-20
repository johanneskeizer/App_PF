# rag_llm.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env.pa"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 128k context by default
DEFAULT_MAX_CTX_CHARS = int(os.getenv("RAG_MAX_CTX_CHARS", "16000"))

# ---- Prompts ----
SYSTEM_STRICT = """You are a precise research assistant.
Answer ONLY using the provided CONTEXT. If the answer is not in the context, say you don't know.
Cite the provided source titles inline like [1], [2]. Be concise and factual.
"""

SYSTEM_BLENDED = """You are a precise research assistant.
Use your general knowledge AND the provided CONTEXT. Treat the CONTEXT as the primary authority for specifics.
- When the context covers the question, base the answer on it and cite source titles inline like [1], [2].
- You may add general knowledge to clarify or connect ideas.
- If the context conflicts with your general knowledge, prefer the context and note the discrepancy briefly.
- If you rely on general knowledge for a detail not present in context, you may include it (no bracketed citation needed).
Be concise and factual.
"""

def _pick_system(mode: str) -> str:
    return SYSTEM_BLENDED if (mode or "").lower() != "strict" else SYSTEM_STRICT

def build_context(matches, max_chars=DEFAULT_MAX_CTX_CHARS):
    """Concatenate retrieved chunks with simple numeric labels and titles."""
    items, total = [], 0
    seen_ids = set()
    for i, m in enumerate(matches, start=1):
        md = (m.metadata or {})
        txt = md.get("text") or ""
        title = md.get("title") or "(untitled)"
        vid = getattr(m, "id", f"vec-{i}")
        if not txt or vid in seen_ids:
            continue
        piece = f"[{i}] {title}\n{txt}\n"
        if total + len(piece) > max_chars:
            break
        items.append(piece)
        total += len(piece)
        seen_ids.add(vid)
    return "\n\n".join(items)

def answer_with_context(question, matches, mode: str = "blended"):
    system = _pick_system(mode)
    context = build_context(matches)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (use as described in your instructions):\n{context if context else '(no context retrieved)'}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

