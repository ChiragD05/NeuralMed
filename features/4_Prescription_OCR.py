import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Prescription OCR",
    "Medicine extraction from handwritten prescriptions"
)

st.info("Module under development")