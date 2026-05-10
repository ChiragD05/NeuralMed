import streamlit as st
from utils.ui import load_css, page_header

load_css()

page_header(
    "Disease Risk Prediction",
    "Predictive analytics using patient health data"
)

st.info("Module under development")