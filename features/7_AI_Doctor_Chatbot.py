import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "AI Doctor Chatbot",
    "RAG-powered conversational medical assistant"
)

st.info("Module under development")