import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Voice Assistant",
    "Speech-enabled healthcare assistant"
)

st.info("Module under development")