import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Medical Image Classification",
    "CNN-based disease prediction from medical images"
)

st.info("Module under development")