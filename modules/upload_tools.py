from pathlib import Path
import streamlit as st
import tempfile
import numpy as np
from openai import OpenAI
from docx import Document
import fitz
from modules import storage
from wordpress_poster import load_wp_config
import requests
from requests.auth import HTTPBasicAuth

def read_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    doc = fitz.open(path)
    text = "\\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\\n".join([p.text for p in doc.paragraphs])

def render_inputs_and_draft(config):
    st.markdown("## 🎛️ Input Panel")

    uploaded_files = st.file_uploader(
        "Upload .txt, .pdf, .docx, .mp3, .wav, .m4a, or .mp4",
        type=["txt", "pdf", "docx", "mp3", "wav", "m4a", "mp4"],
        accept_multiple_files=True
    )

    if uploaded_files:
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        for file in uploaded_files:
            try:
                if file.type.startswith("audio") or file.name.endswith((".mp3", ".wav", ".m4a", ".mp4")):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    with open(tmp_path, "rb") as f:
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                    st.session_state["draft"] = transcript.text
                    st.success(f"✅ Transcribed: {file.name}")
                elif file.type == "text/plain":
                    st.session_state["draft"] = file.read().decode("utf-8")
                    st.success(f"📄 Processed: {file.name}")
                elif file.type == "application/pdf":
                    st.session_state["draft"] = read_pdf(file)
                    st.success(f"📄 Processed: {file.name}")
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    st.session_state["draft"] = read_docx(file)
                    st.success(f"📄 Processed: {file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

    if "draft" not in st.session_state:
        st.session_state["draft"] = ""

    st.text_area("✍️ Draft Input", value=st.session_state["draft"], height=300, key="draft")
    col1, col2 = st.columns(2)
    with col1:
                if st.button("🧼 Clear Draft Input"):
                    st.session_state["draft"] = ""
                    st.session_state["revised"] = ""
                    st.rerun()

    with col2:
                if st.button("🧽 Clear All Memory"):
                    st.session_state.clear()
                    st.rerun()


    tone = st.text_input("🎭 Preferred writing tone", value="Spoken, slightly ironic, but grammatically correct")

    if st.button("🪄 Revise with GPT"):
        with st.spinner("💬 Revising with GPT..."):
            try:
                client = OpenAI(api_key=config["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"Rewrite the following in the tone: {tone}"},
                        {"role": "user", "content": st.session_state["draft"]}
                    ]
                )
                st.session_state["revised"] = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"❌ GPT revision failed: {e}")

    if "revised" not in st.session_state:
        st.session_state["revised"] = ""

    st.text_area("📝 Revised Output", value=st.session_state["revised"], height=300, key="revised")

    storage.save_controls(st.session_state["draft"], st.session_state["revised"], config)

    with st.expander("📤 Post to WordPress", expanded=False):
        wp_cfg = load_wp_config("wp_config.yaml")

        base_url = st.text_input("Base URL", value=wp_cfg["WP_BASE_URL"], key="wp_base_url")
        username = st.text_input("Username", value=wp_cfg["WP_USERNAME"], key="wp_user")
        password = st.text_input("App Password", type="password", value=wp_cfg["WP_APP_PASSWORD"], key="wp_pass", autocomplete="off")
        title = st.text_input("Post Title", value=wp_cfg["DEFAULT_TITLE"], key="wp_title")

        if st.button("🚀 Publish to WordPress", key="post_wp_button"):
            if not st.session_state.get("revised"):
                st.warning("⚠️ No revised text to post.")
            else:
                try:
                    post_url = base_url.rstrip("/") + "/wp-json/wp/v2/posts"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "title": title,
                        "content": st.session_state["revised"],
                        "status": wp_cfg.get("POST_STATUS", "draft")
                    }
                    response = requests.post(post_url, json=payload, auth=HTTPBasicAuth(username, password), headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    st.success(f"✅ Post created: {result.get('link')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to post to WordPress: {e}")

