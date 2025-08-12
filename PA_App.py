import os
import io
import json
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Local modules
from embed import embed, embed_with
from pine import upsert, make_id, query as query_pa, query_index, has_index, dims_summary
from rag_llm import answer_with_context
from ingest_utils import read_file

# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------
load_dotenv(Path(__file__).with_name(".env.pa"))

st.set_page_config(page_title="Personal Assistant • One Pinecone Index", layout="wide")
st.title("Personal Assistant • One Pinecone Index")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show_json(obj):
    """Robust JSON viewer that never crashes the UI."""
    if obj is None:
        st.info("No data")
        return
    try:
        if isinstance(obj, (dict, list)):
            st.code(json.dumps(obj, indent=2, ensure_ascii=False), language="json")
            return
        for attr in ("to_dict", "model_dump", "dict"):
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                d = getattr(obj, attr)()
                st.code(json.dumps(d, indent=2, ensure_ascii=False), language="json")
                return
        if isinstance(obj, (str, bytes)):
            s = obj if isinstance(obj, str) else obj.decode("utf-8", "ignore")
            try:
                st.code(json.dumps(json.loads(s), indent=2, ensure_ascii=False), language="json")
                return
            except Exception:
                st.code(s)
                return
        st.code(str(obj))
    except Exception as e:
        st.warning(f"Could not render JSON: {e}")


# ----------------------------------------------------------------------------
# Sidebar: light controls only
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Controls")
    namespace = st.selectbox("Namespace", ["health", "travel", "finance", "writing", "cpa"])  # cpa included
    include_pd = st.checkbox("Also use PD-Knowledge", value=(namespace == "health"))
    top_k = st.slider("Top K per index", 3, 20, 8)
    strict_mode = st.checkbox("Strict RAG (context only)", value=False)

    with st.expander("Embedding / Index diagnostics"):
        try:
            show_json(dims_summary())
        except Exception as e:
            st.info(f"Diagnostics unavailable: {e}")

# Models from env
PA_EMBED_MODEL = os.getenv("PA_EMBED_MODEL", "text-embedding-3-large")   # 3072-d
PD_EMBED_MODEL = os.getenv("PD_EMBED_MODEL", "text-embedding-3-small")   # 1536-d

# ----------------------------------------------------------------------------
# Main area: Tabs
# ----------------------------------------------------------------------------
tab_ingest, tab_search, tab_logs, tab_cpa = st.tabs(["Ingest", "Search & Ask", "Logs", "CPA"])

# -----------------------------
# Ingest tab
# -----------------------------
with tab_ingest:
    st.subheader("Ingest content into PA (PD-Knowledge is read-only)")
    mode = st.radio("Mode", ["Text", "Files", "Numbers"], horizontal=True)

    # TEXT MODE
    if mode == "Text":
        with st.form("ingest_text"):
            doc_type = st.selectbox("Type", ["note", "doc", "record", "web", "rss", "transcript"])
            title = st.text_input("Title")
            tags = st.text_input("Tags (comma-separated)")
            text = st.text_area("Paste text", height=220)
            submit = st.form_submit_button("Embed & Upsert")
        if submit:
            if not title.strip() or not text.strip():
                st.error("Please provide Title and Text.")
            else:
                try:
                    vec = embed(text)[0]
                    vid = make_id(namespace, title)
                    meta = {
                        "assistant": namespace,
                        "type": doc_type,
                        "title": title.strip(),
                        "source": "manual",
                        "url": "",
                        "created_at": time.strftime("%Y-%m-%d"),
                        "tags": [t.strip() for t in tags.split(",") if t.strip()],
                        "lang": "en",
                        "text": text,
                    }
                    upsert(namespace, [{"id": vid, "values": vec, "metadata": meta}])
                    st.success(f"Ingested → {vid}")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

    # FILES MODE
    elif mode == "Files":
        st.caption("Accepted: .txt, .pdf (OCR if needed), .rtf, .csv")
        doc_type = st.selectbox("Type", ["doc", "record", "transcript", "web"], key="file_doctype")
        tags = st.text_input("Tags (comma-separated)", key="file_tags")
        files = st.file_uploader("Upload", type=["txt", "pdf", "rtf", "csv"], accept_multiple_files=True)
        if st.button("Embed & Upsert files"):
            if not files:
                st.info("Choose one or more files.")
            else:
                done, failed = 0, 0
                for f in files:
                    try:
                        docs = read_file(f.name, f.read())  # list[(title, text)]
                        for (t, text) in docs:
                            if not text.strip():
                                continue
                            vec = embed(text)[0]
                            vid = make_id(namespace, t)
                            meta = {
                                "assistant": namespace,
                                "type": doc_type,
                                "title": t,
                                "source": "file",
                                "filename": f.name,
                                "url": "",
                                "created_at": time.strftime("%Y-%m-%d"),
                                "tags": [s.strip() for s in tags.split(",") if s.strip()],
                                "lang": "en",
                                "text": text,
                            }
                            upsert(namespace, [{"id": vid, "values": vec, "metadata": meta}])
                            done += 1
                    except Exception as e:
                        failed += 1
                        st.error(f"{f.name}: {e}")
                st.success(f"Ingested chunks: {done} · Failed files: {failed}")

    # NUMBERS MODE
    # In your Ingest tab, swap the Numbers branch:
    elif mode == "Numbers":
    # make selected namespace visible to the form/shim
        st.session_state["namespace"] = namespace
        from numeric_input_form import render_numeric_input_form
        render_numeric_input_form()

# -----------------------------
# Search & Ask tab (Blended RAG)
# -----------------------------
with tab_search:
    st.subheader("Search & Ask (blended RAG)")
    q = st.text_input("Your question")
    ask = st.button("Run")
    if ask:
        if not q.strip():
            st.info("Type a question.")
        else:
            try:
                # 1) Embed & query PA
                qvec_pa = embed_with(PA_EMBED_MODEL, q)[0]
                res_pa = query_pa(namespace, qvec_pa, top_k=top_k)
                matches = list(getattr(res_pa, "matches", []) or [])

                # 2) Optionally include PD-Knowledge (read-only)
                if include_pd and has_index("pd-knowledge"):
                    qvec_pd = embed_with(PD_EMBED_MODEL, q)[0]
                    res_pd = query_index("pd-knowledge", None, qvec_pd, top_k=top_k)
                    matches += list(getattr(res_pd, "matches", []) or [])
                elif include_pd:
                    st.info("pd-knowledge index not found — skipping.")

                # 3) Sort & display sources
                matches.sort(key=lambda m: getattr(m, "score", 0.0), reverse=True)
                st.markdown("**Sources:**")
                if not matches:
                    st.write("No results.")
                else:
                    for i, m in enumerate(matches[:max(top_k, 10)], start=1):
                        md = m.metadata or {}
                        ns = getattr(m, "namespace", namespace) or namespace
                        st.write(f"{i}. **{md.get('title','(untitled)')}** — {m.score:.3f} · ns: `{ns}` · id: `{m.id}`")

                # 4) Ask GPT with blended context
                if matches:
                    mode = "strict" if strict_mode else "blended"
                    ans = answer_with_context(q, matches, mode=mode)
                    st.markdown("---")
                    st.markdown("**Answer:**")
                    st.write(ans)
            except Exception as e:
                st.error(f"Search failed: {e}")

# -----------------------------
# Logs tab (optional viewer if you use log.py elsewhere)
# -----------------------------
with tab_logs:
    st.subheader("Ingestion Log")
    try:
        import pandas as pd
        p = Path("logs/ingest_log.csv")
        if p.exists():
            try:
                df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
                expected = [
                    "ts_utc","namespace","type","title","vector_id",
                    "content_hash","chars","model","status","error",
                    "source","filename"
                ]
                for col in expected:
                    if col not in df.columns:
                        df[col] = ""
                df = df[[c for c in expected if c in df.columns]]
                st.dataframe(df.tail(200), use_container_width=True)
                st.download_button("Download CSV", data=p.read_bytes(), file_name="ingest_log.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"Couldn't parse log: {e}")
                st.code(p.read_text(encoding="utf-8", errors="ignore"))
        else:
            st.info("No logs yet.")
    except Exception as e:
        st.info("Pandas not available for log viewing.")

# -----------------------------
# CPA tab — stub hook
# -----------------------------
with tab_cpa:
    st.subheader("Content Production Assistant")
    try:
        from modules import config_loader, upload_tools, ui_layout
        cfg = config_loader.load_config()          # merges .env.pa and config.yaml [6]
        if not config_loader.authenticate():       # optional password gate [6]
            st.stop()
        ui_layout.show_header()                    # nice header [5]
        upload_tools.render_inputs_and_draft(cfg)  # upload txt/pdf/docx/audio/video, draft+revise [3]
        ui_layout.show_footer()                    # footer [5]
    except Exception as e:
        st.error(f"CPA failed to load: {e}")
