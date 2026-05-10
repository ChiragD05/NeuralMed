import streamlit as st
from utils.config import DISCLAIMER

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def show_disclaimer():
    st.warning(DISCLAIMER)

def page_header(title, subtitle=""):
    st.title(title)

    if subtitle:
        st.caption(subtitle)

    st.divider()

def feature_card(title, description):
    st.markdown(
        f"""
        ### {title}
        {description}
        """
    )