import streamlit as st
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone
from fpdf import FPDF
import io
import os

def save_controls(draft, revised, config):
    text = revised or draft
    if not text:
        st.warning("Nothing to save.")
        return

    st.markdown("### 📦 Save Options")

    col1, col2 = st.columns(2)

    with col1:
        pinecone_connected = st.checkbox("Pinecone connected?", key="pinecone_confirm")
        st.button(
            "📌 Save to Pinecone",
            disabled=not pinecone_connected,
            on_click=lambda: save_to_pinecone(text, config)
        )

    with col2:
        file_format = st.selectbox("📁 Choose file format", ["HTML", "TXT", "PDF"], key="download_format")

        now = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename = f"CPA_{now}.{file_format.lower()}"

        if file_format == "HTML":
            content = f"<html><body><pre>{text}</pre></body></html>"
            mime = "text/html"
            data = content.encode("utf-8")

        elif file_format == "TXT":
            content = text
            mime = "text/plain"
            data = content.encode("utf-8")

        elif file_format == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
            if not os.path.exists(font_path):
                st.error("⚠️ Missing DejaVuSans.ttf font file in the modules folder.")
                return

            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", "", 12)

            for line in text.split("\n"):
                pdf.multi_cell(0, 10, line)

            try:
                pdf_output = pdf.output(dest='S').encode('latin1')
                data = pdf_output
                mime = "application/pdf"
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
                return

        st.download_button(
            label=f"💾 Download as {file_format}",
            file_name=filename,
            mime=mime,
            data=data,
        )

def save_to_pinecone(text, config):
    try:
        vector_label = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        pc = Pinecone(api_key=config["PINECONE_API_KEY"])
        index = pc.Index(config["PINECONE_INDEX_NAME"])
        index.upsert([{
            "id": vector_label,
            "values": embedding,
            "metadata": {
                "source": "writing_assistant",
                "timestamp": datetime.now().isoformat(),
                "text": text
            }
        }])
        st.success(f"✅ Uploaded as {vector_label}")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

