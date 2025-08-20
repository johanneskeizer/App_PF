# === modules/ui_layout.py ===
import streamlit as st

def show_header():
    st.markdown("""
        <h1 style='font-size: 2rem; margin-bottom: 0;'>🧠 Content Production Assistant</h1>
        <hr style='margin-top: 0.5rem; margin-bottom: 1rem;' />
    """, unsafe_allow_html=True)

def show_footer():
    st.markdown("<small>v1.0 — Powered by GPT and Pinecone</small>", unsafe_allow_html=True)
