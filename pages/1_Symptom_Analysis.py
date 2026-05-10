import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Symptom Analysis",
    "AI-powered symptom understanding and triage"
)

st.info("Module under development")