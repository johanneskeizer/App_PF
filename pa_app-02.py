import os
import io
import json
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

if "view" not in st.session_state:
    st.session_state["view"] = "tabs"

NS="pa"

def _pa_query_direct(vec, top_k=8, namespace=None, include_metadata=True):
    if not pa_index:
        raise RuntimeError("pa_index not available")
    dim = _index_dimension(pa_index, 3072)
    v = list(vec or [])
    if len(v) < dim:
        v = v + [0.0] * (dim - len(v))
    elif len(v) > dim:
        v = v[:dim]
    ns = NS if (namespace is None or namespace == "") else namespace
    return pa_index.query(vector=v, top_k=top_k, namespace=ns, include_metadata=include_metadata)


# ---- UIG bridge (ingestion only) ----
from pa_uig_bridge import (
    trigger_uig_with_text,
    trigger_uig_on_folder,
    save_uploaded_files_to_staging,
)

# ---- Read-only retrieval deps (optional, guarded) ----
try:
    from pine import query as query_pa, query_index, has_index, pa_index, dims_summary
except Exception:
    query_pa = query_index = has_index = pa_index = None
    dims_summary = None
try:
    from embed import embed_with  # for query embedding
except Exception:
    embed_with = None
try:
    from rag_llm import answer_with_context
except Exception:
    answer_with_context = None

# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------
load_dotenv(Path(__file__).with_name(".env.pa"))

# ----------------------------------------------------------------------------
# Helpers





def _index_dimension(idx, default: int = 3072) -> int:
    """Get index dim robustly; fall back to default."""
    try:
        stats = idx.describe_index_stats()
        if isinstance(stats, dict):
            dim = stats.get("dimension")
        else:
            dim = getattr(stats, "dimension", None)
        return int(dim) if dim else default
    except Exception:
        return default
    
def _chunk_text(md):
    for k in ("text","chunk","content"):
        if md.get(k):
            return str(md[k])
    if md.get("filename") and md.get("path"):
        return f"[File] {md['filename']} — {md['path']}"
    return ""

    

def _unit_probe(idx_dim: int) -> list[float]:
    v = [0.0] * idx_dim
    v[0] = 1.0
    return v

def render_physio_report():
    st.subheader("Physiological Metrics Report")
    if not pa_index:
        st.info("Pinecone 'pa' index handle not available in this build.")
        return

    pc_dim = _index_dimension(pa_index, default=3072)
    probe = _unit_probe(pc_dim)

    # Try explicit namespace; if somehow empty, retry without
    try:
        results = pa_index.query(vector=probe, top_k=2000, namespace=NS, include_metadata=True)
        matches = list(getattr(results, "matches", []) or [])
        if not matches:
            results = pa_index.query(vector=probe, top_k=2000, include_metadata=True)
            matches = list(getattr(results, "matches", []) or [])
    except Exception as e:
        st.error(f"Query failed: {e}")
        return

    if not matches:
        st.info("No physiological records found.")
        return

    rows = []
    for m in matches:
        md = (m.metadata or {}).copy()
        md["id"] = m.id
        md["date"] = str(md.get("date", md.get("created_at", "")))
        # normalize tags for filtering
        tags = md.get("tags", [])
        if isinstance(tags, str):
            try:
                # sometimes stored as JSON string; try to parse
                import json as _json
                tags = _json.loads(tags)
            except Exception:
                tags = [t.strip() for t in tags.split(",") if t.strip()]
        md["_tags_list"] = tags
        rows.append(md)

    import pandas as pd
    df = pd.DataFrame(rows)

    # Keep only rows tagged as metrics (or with explicit type set)
    def _has_tag(x, tag): 
        try: return tag in (x or [])
        except: return False
    mask = df["_tags_list"].apply(lambda x: _has_tag(x, "metrics")) | (df.get("type", "") == "prefilled_metrics")
    df = df[mask] if mask.any() else df  # if no tags yet, show all to help debugging

    c1, c2 = st.columns(2)
    with c1:
        min_date = st.date_input("Start date (optional)", value=None, key="phys_min_date")
    with c2:
        max_date = st.date_input("End date (optional)", value=None, key="phys_max_date")

    if "date" in df.columns:
        if min_date: df = df[df["date"] >= str(min_date)]
        if max_date: df = df[df["date"] <= str(max_date)]

    if df.empty:
        st.info("No rows after filtering.")
        return

    df = df.sort_values("date", ascending=False, na_position="last")
    phys_cols = [
        "date","weight","sleep","steps","systolic","diastolic",
        "glucose_min","glucose_max","heartbeat_min","heartbeat_med","heartbeat_max",
        "cholesterol","ldl","hdl","triglycerides","oxygen_saturation","id",
    ]
    use_cols = [c for c in phys_cols if c in df.columns] or list(df.columns)
    st.dataframe(df[use_cols], use_container_width=True)
    st.download_button("Download as CSV", df[use_cols].to_csv(index=False),
                       file_name="metrics_report.csv", mime="text/csv")


def render_symptoms_report():
    st.subheader("PD Symptoms Report")
    if not pa_index:
        st.info("Pinecone 'pa' index handle not available in this build.")
        return

    pc_dim = _index_dimension(pa_index, default=3072)
    probe = _unit_probe(pc_dim)

    try:
        results = pa_index.query(vector=probe, top_k=2000, namespace=NS, include_metadata=True)
        matches = list(getattr(results, "matches", []) or [])
        if not matches:
            results = pa_index.query(vector=probe, top_k=2000, include_metadata=True)
            matches = list(getattr(results, "matches", []) or [])
    except Exception as e:
        st.error(f"Query failed: {e}")
        return

    if not matches:
        st.info("No symptom logs found.")
        return

    rows = []
    for m in matches:
        md = m.metadata or {}
        rec = {"id": m.id}
        rec["date"] = str(md.get("date", md.get("created_at", "")))
        # tags for filtering
        tags = md.get("tags", [])
        if isinstance(tags, str):
            try:
                import json as _json
                tags = _json.loads(tags)
            except Exception:
                tags = [t.strip() for t in tags.split(",") if t.strip()]
        rec["_tags_list"] = tags
        # collect symptom fields if present
        for k, v in md.items():
            if str(k).startswith("sym_"):
                rec[k] = v
        if "notes" in md:
            rec["notes"] = md["notes"]
        rows.append(rec)

    import pandas as pd
    df = pd.DataFrame(rows)

    # Keep only rows tagged as symptoms (or explicit type)
    def _has_tag(x, tag): 
        try: return tag in (x or [])
        except: return False
    mask = df["_tags_list"].apply(lambda x: _has_tag(x, "symptoms")) | (df.get("type", "") == "symptom_log")
    df = df[mask] if mask.any() else df

    c3, c4 = st.columns(2)
    with c3:
        s_min_date = st.date_input("Start date (optional)", value=None, key="sym_min_date")
    with c4:
        s_max_date = st.date_input("End date (optional)", value=None, key="sym_max_date")

    if "date" in df.columns:
        if s_min_date: df = df[df["date"] >= str(s_min_date)]
        if s_max_date: df = df[df["date"] <= str(s_max_date)]

    if df.empty:
        st.info("No rows after filtering.")
        return

    df = df.sort_values("date", ascending=False, na_position="last")
    sym_cols = sorted([c for c in df.columns if c.startswith("sym_")])
    use_cols = ["date"] + sym_cols + (["notes"] if "notes" in df.columns else []) + ["id"]
    use_cols = [c for c in use_cols if c in df.columns] or list(df.columns)
    st.dataframe(df[use_cols], use_container_width=True)
    st.download_button("Download as CSV", df[use_cols].to_csv(index=False),
                       file_name="symptoms_report.csv", mime="text/csv")


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
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    if st.button("Show Physiological Data Report"):
        st.session_state["view"] = "report_physio"
        st.rerun()

    st.markdown("---")
    if st.button("Show PD Symptoms Report"):
        st.session_state["view"] = "report_symptoms"
        st.rerun()

    
    st.markdown("---")
    st.subheader("Search Controls")
    include_pd = st.checkbox("Also use PD-Knowledge", value=False)
    top_k = st.slider("Top K per index", 3, 20, 8)
    strict_mode = st.checkbox("Strict RAG (context only)", value=False)


    with st.expander("Embedding / Index diagnostics"):
      try:
        if dims_summary:
            info = dims_summary()  # expect dict-like
            st.write("**Indexes**")
            if isinstance(info, dict):
                for name, meta in info.items():
                    st.write(f"- `{name}` · dim: {meta.get('dimension','?')} · metric: {meta.get('metric','?')} · size: {meta.get('count','?')}")
            else:
                st.json(info)  # fallback
        else:
            st.caption("Diagnostics unavailable in this build.")
      except Exception as e:
        st.caption(f"Diagnostics error: {e}")


# Models from env (for SEARCH only; ingestion is handled by UIG)
PA_EMBED_MODEL = os.getenv("PA_EMBED_MODEL", "text-embedding-3-large")   # 3072-d
PD_EMBED_MODEL = os.getenv("PD_EMBED_MODEL", "text-embedding-3-small")   # 1536-d

# ----------------------------------------------------------------------------
# Main area: Tabs
# ----------------------------------------------------------------------------

# If a report view is selected, show that screen and return early.
if st.session_state["view"] == "report_physio":
    st.header("Physiological Metrics Report")
    st.button("← Back to main", on_click=lambda: st.session_state.update(view="tabs"))
    # (render your full physio report here, no tabs, default: show all data)
    render_physio_report()  # ← move your existing physio-report logic into this function
    st.stop()

if st.session_state["view"] == "report_symptoms":
    st.header("PD Symptoms Report")
    st.button("← Back to main", on_click=lambda: st.session_state.update(view="tabs"))
    # (render your full symptoms report here, no tabs, default: show all data)
    render_symptoms_report()  # ← move your symptoms-report logic into this function
    st.stop()

# Otherwise, show the normal tabs UI
tab_ingest, tab_search, tab_logs, tab_cpa = st.tabs(["Ingest", "Search & Ask", "Logs", "CPA"])

# -----------------------------
# Ingest tab (all writes go through UIG)
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
                    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
                    staging, rc = trigger_uig_with_text(
                        title=title.strip(),
                        text=text,
                        tags=tags_list,
                        dry_run=False,
                    )
                    if rc == 0:
                        st.success(f"Ingest queued via UIG ✓ (staging: {staging})")
                    else:
                        st.error(f"UIG returned code {rc} (staging: {staging})")
                except Exception as e:
                    st.error(f"UIG trigger failed: {e}")


    # FILES MODE
    elif mode == "Files":
        st.caption("Accepted: .txt, .md, .pdf, .rtf, .csv")
        doc_type = st.selectbox("Type", ["doc", "record", "transcript", "web"], key="file_doctype")
        tags = st.text_input("Tags (comma-separated)", key="file_tags")
        files = st.file_uploader("Upload", type=["txt", "md", "pdf", "rtf", "csv"], accept_multiple_files=True)
        if st.button("Embed & Upsert files"):
            
            if not files:
                st.info("Choose one or more files.")
            else:
                try:
                    staging = save_uploaded_files_to_staging(files)
                    tags_list = [s.strip() for s in tags.split(",") if s.strip()]
                    rc = trigger_uig_on_folder(staging, tags=tags_list, dry_run=False)
                    if rc == 0:
                        st.success(f"Ingested via UIG ✓ (staging: {staging})")
                    else:
                        st.error(f"UIG returned code {rc} (staging: {staging})")
                except Exception as e:
                    st.error(f"UIG trigger failed: {e}")


    # NUMBERS MODE
    elif mode == "Numbers":
        # ------ A) Manual Numbers (optional external form) ------
        with st.expander("Manual Numeric Input (fully flexible)", expanded=False):
            try:
                from numeric_input_form import render_numeric_input_form
                render_numeric_input_form()
            except Exception as e:
                st.error(f"Numeric form failed: {e}")

        # ------ B) Prefilled editable Data Input Sheet ------
        with st.expander("Prefilled Data Input Sheet", expanded=True):
            with st.form("prefilled_daily_metrics"):
                st.write("**Person:** Johannes Keizer")
                date = st.date_input("Date", value=None)
                weight = st.number_input("Weight (kg)", 0.0, 200.0, 73.0, step=0.1, format="%.1f")
                sleep = st.number_input("Sleep (hours)", 0.0, 24.0, 5.0, step=0.1, format="%.1f")
                steps = st.number_input("Steps (km)", 0.0, 30.0, 5.0, step=0.1, format="%.1f")
                st.write("**Blood Pressure**")
                systolic = st.number_input("Systolic (mmHg)", 0, 250, 120)
                diastolic = st.number_input("Diastolic (mmHg)", 0, 150, 80)
                glucose_min = st.number_input("Glucose min (mmol/l)", 0.0, 20.0, 4.0, step=0.1)
                glucose_max = st.number_input("Glucose max (mmol/l)", 0.0, 20.0, 7.0, step=0.1)
                heartbeat_min = st.number_input("Heartbeat min (bpm)", 0, 250, 45)
                heartbeat_med = st.number_input("Heartbeat med (bpm)", 0, 250, 67)
                heartbeat_max = st.number_input("Heartbeat max (bpm)", 0, 250, 150)
                cholesterol = st.number_input("Cholesterol (mmol/l)", 0.0, 20.0, 4.0, step=0.1)
                ldl = st.number_input("LDL (mmol/l)", 0.0, 20.0, 0.0, step=0.1)
                hdl = st.number_input("HDL (mmol/l)", 0.0, 20.0, 0.0, step=0.1)
                triglycerides = st.number_input("Triglycerides (mmol/l)", 0.0, 10.0, 1.2, step=0.1)
                oxygen_sat = st.number_input("Oxygen Saturation (%)", 0, 100, 98)
                submit_prefilled = st.form_submit_button("Submit Prefilled Metrics")

            


                if submit_prefilled:
                    rec = {
                        "person": "Johannes Keizer",
                        "date": str(date),
                        "weight": weight,
                        "sleep": sleep,
                        "steps": steps,
                        "systolic": systolic,
                        "diastolic": diastolic,
                        "glucose_min": glucose_min,
                        "glucose_max": glucose_max,
                        "heartbeat_min": heartbeat_min,
                        "heartbeat_med": heartbeat_med,
                        "heartbeat_max": heartbeat_max,
                        "cholesterol": cholesterol,
                        "ldl": ldl,
                        "hdl": hdl,
                        "triglycerides": triglycerides,
                        "oxygen_saturation": oxygen_sat,
                    }
                    # short textual summary for readability
                    summary = (
                        f"Daily Metrics {rec['date']}: BP {systolic}/{diastolic} mmHg; "
                        f"weight {weight} kg; sleep {sleep} h; steps {steps} km; "
                        f"Glu {glucose_min}-{glucose_max} mmol/l; HR {heartbeat_min}/{heartbeat_med}/{heartbeat_max} bpm; "
                        f"Chol {cholesterol} mmol/l; LDL {ldl}; HDL {hdl}; TG {triglycerides}; SpO2 {oxygen_sat}%."
                    )
                    staging, rc = trigger_uig_with_text(
                        title=f"Daily_Metrics_{rec['date']}",
                        text=summary,
                        tags=["metrics","prefilled"],
                        meta_type="prefilled_metrics",
                        extra_metadata=rec,           # <- structured fields go into Pinecone metadata
                        dry_run=False,
                    )

                    if rc == 0:
                            st.info(f"Metrics sent to UIG ✓ (staging: {staging})")
                    else:
                            st.warning(f"UIG returned code {rc} (staging: {staging})")
                    
           

        # ------ C) Parkinson’s Symptom Log (twice monthly) ------
        with st.expander("Parkinson’s Symptom Log", expanded=True):
            with st.form("parkinsons_symptom_log"):
                s_date = st.date_input("Date", value=None, key="symptoms_date")

                SYMPTOMS = [
                    "Tremor right side","Tremor left side","Stiffness legs","Stiffness hands",
                    "Prone to fall","Gait changes","Defecation (constipation)","Urination problems",
                    "Salivating (drooling)","Speech changes","Swallowing difficulties","Reduced facial expression",
                    "Decreased blinking","Micrographia (small handwriting)","Sleep disturbances","Loss of smell",
                    "Depression","Apathy","Anxiety","Hallucinations","Fatigue","Pain","Sexual dysfunction",
                ]

                severities = {}
                cols = st.columns(3)
                for i, name in enumerate(SYMPTOMS):
                    with cols[i % 3]:
                        severities[name] = st.number_input(
                            f"{name} (0–10)", min_value=0, max_value=10, value=0, step=1, key=f"sev_{i}"
                        )

                notes = st.text_area("Optional notes", height=80, key="symptoms_notes")
                submit_symptoms = st.form_submit_button("Submit Symptom Log")

            
              
       
            
        if submit_symptoms:
            date_str = str(s_date)
            extra = {"date": date_str}
            # Build severities as sym_* fields so reports can pick them up
            for k, v in severities.items():
                extra["sym_" + k.replace(" ", "_").lower()] = v
            if notes.strip():
                extra["notes"] = notes.strip()

            summary = (
                f"Parkinson’s Symptom Log for {date_str}. "
                f"Severities (0–10): " + ", ".join([f"{k}: {v}" for k, v in severities.items()]) +
                (f" Notes: {notes.strip()}" if notes.strip() else "")
            )

            staging, rc = trigger_uig_with_text(
                title=f"Symptoms_{date_str}",
                text=summary,
                tags=["symptoms","parkinsons"],
                meta_type="symptom_log",
                extra_metadata=extra,      # <- structured fields
                dry_run=False,
            )





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
                if not (embed_with and pa_index):
                    raise RuntimeError("Search modules not available in this build.")

                # 1) Embed & query PA
                qvec_pa = embed_with(PA_EMBED_MODEL, q)[0]
                res_pa = _pa_query_direct(qvec_pa, top_k=top_k, namespace=NS)
                matches = list(getattr(res_pa, "matches", []) or [])

                # 2) Optionally include PD-Knowledge
                if include_pd and has_index and has_index("pd-knowledge"):
                    qvec_pd = embed_with(PD_EMBED_MODEL, q)[0]
                    # use the same padding trick for pd index
                    res_pd = query_index("pd-knowledge", None, qvec_pd, top_k=top_k)
                    matches += list(getattr(res_pd, "matches", []) or [])
                elif include_pd:
                    st.info("pd-knowledge index not found — skipping.")

                # 3) Sort & display sources
                matches.sort(key=lambda m: getattr(m, "score", 0.0), reverse=True)
                st.markdown("**Sources:**")
                if not matches:
                    st.write("No results.")
                       
                for i, m in enumerate(matches[:max(top_k, 10)], start=1):
                    md = m.metadata or {}
                    st.write(f"{i}. **{md.get('title','(untitled)')}** — {m.score:.3f} · ns: `{NS}` · id: `{m.id}`")

                # 4) Build context text
                def _chunk_text(md):
                    # common keys we might have stored
                    for k in ("text","chunk","content"):
                        if md.get(k):
                            return str(md[k])
                    # last resort: filename or brief JSON of metadata
                    if md.get("filename") and md.get("path"):
                        return f"[File] {md['filename']} — {md['path']}"
                    return ""
                top_ctx = [_chunk_text(m.metadata or {}) for m in matches[:top_k]]
                top_ctx = [t for t in top_ctx if t]
                context_text = "\n\n---\n\n".join(top_ctx) if top_ctx else ""

                # 5) Answer
                if answer_with_context and context_text:
                    mode = "strict" if strict_mode else "blended"
                    ans = answer_with_context(q, matches, mode=mode)
                else:
                    # Simple fallback using OpenAI directly if needed
                    from openai import OpenAI
                    client = OpenAI()
                    prompt = f"Question:\n{q}\n\nContext:\n{context_text}\n\nAnswer using only the provided context."
                    chat = client.chat.completions.create(
                        model=os.getenv("PA_CHAT_MODEL","gpt-4o-mini"),
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.2,
                    )
                    ans = chat.choices[0].message.content

                st.markdown("---")
                st.markdown("**Answer:**")
                st.write(ans)

            except Exception as e:
                st.error(f"Search failed: {e}")

# -----------------------------
# Logs tab (optional viewer)
# -----------------------------
with tab_logs:
    st.subheader("Ingestion Log")
    try:
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
    except Exception:
        st.info("Pandas not available for log viewing.")

# -----------------------------
# CPA tab — stub hook
# -----------------------------
with tab_cpa:
    st.subheader("Content Production Assistant")
    try:
        from modules import config_loader, upload_tools, ui_layout
        cfg = config_loader.load_config()
        if not config_loader.authenticate():
            st.stop()
        ui_layout.show_header()
        upload_tools.render_inputs_and_draft(cfg)
        ui_layout.show_footer()
    except Exception as e:
        st.error(f"CPA failed to load: {e}")
