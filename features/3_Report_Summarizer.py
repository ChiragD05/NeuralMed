import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Medical Report Summarizer",
    "Transformer-based report analysis"
)

st.info("Module under development")