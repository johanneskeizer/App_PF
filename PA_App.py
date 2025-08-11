# PA_App.py
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from embed import embed
from pine import upsert, query, make_id

# load .env.pa that sits next to this file
load_dotenv(Path(__file__).with_name(".env.pa"))

st.set_page_config(page_title="Personal Assistant", layout="wide")
st.title("Personal Assistant • One Pinecone Index")

with st.sidebar:
    st.subheader("Workspace")
    namespace = st.selectbox("Namespace", ["health", "travel", "finance", "writing"])

    st.markdown("---")
    st.subheader("Ingest")
    doc_type = st.selectbox("Type", ["note", "doc", "record", "web", "rss", "transcript"])
    title = st.text_input("Title")
    tags = st.text_input("Tags (comma-separated)")
    text = st.text_area("Paste text to ingest", height=160)

    if st.button("Embed & Upsert", type="primary", use_container_width=True):
        if not title.strip() or not text.strip():
            st.error("Please provide both Title and Text.")
        else:
            vec = embed(text)[0]
            vid = make_id(namespace, title)
            meta = {
                "assistant": namespace,
                "type": doc_type,
                "title": title.strip(),
                "source": "manual",
                "url": "",
                "created_at": __import__("time").strftime("%Y-%m-%d"),
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "lang": "en",
            }
            upsert(namespace, [{"id": vid, "values": vec, "metadata": meta}])
            st.success(f"Ingested → {vid}")

st.markdown("### Search")
q = st.text_input("Semantic query")
if st.button("Search", type="primary"):
    if not q.strip():
        st.info("Type something to search.")
    else:
        qvec = embed(q)[0]
        res = query(namespace, qvec, top_k=8)
        if not getattr(res, "matches", []):
            st.write("No results.")
        else:
            for m in res.matches:
                st.write(f"**{m.metadata.get('title','(untitled)')}** — {m.score:.3f}")
                st.caption(f"{m.id} · {m.metadata.get('type')} · {', '.join(m.metadata.get('tags',[]))}")
                st.write("---")
