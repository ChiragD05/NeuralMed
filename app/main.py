import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from streamlit_option_menu import option_menu

from utils.ui import load_css, show_disclaimer
from utils.config import APP_NAME, APP_TAGLINE

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

with st.sidebar:
    st.markdown("## 🩺 NeuralMed AI")
    st.caption("Medical decision support dashboard")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Dashboard",
            "Symptom Analysis",
            "Medical Imaging",
            "Report Summarizer",
            "Prescription OCR",
            "Risk Prediction",
            "Voice Assistant",
            "AI Chatbot",
        ],
        icons=[
            "house",
            "activity",
            "image",
            "file-earmark-text",
            "camera",
            "heart-pulse",
            "mic",
            "chat-dots",
        ],
        default_index=0,
    )

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.4rem;">AI-Powered Medical Decision Support Assistant</h1>
        <p style="margin-top:10px; font-size:1.05rem; opacity:0.95;">
            A multimodal healthcare assistant for symptoms, prescriptions, reports, medical images,
            voice input, and AI-guided triage.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Symptom Analysis</div>
            <div class="metric-value">NLP</div>
            <div class="metric-subtitle">Triage-style guidance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Medical Imaging</div>
            <div class="metric-value">CNN</div>
            <div class="metric-subtitle">Image classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Medical Reports</div>
            <div class="metric-value">NLP + LLM</div>
            <div class="metric-subtitle">Summaries in plain English</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">AI Chatbot</div>
            <div class="metric-value">RAG</div>
            <div class="metric-subtitle">Grounded medical answers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("What this assistant can do")
st.markdown(
    """
    - Understand symptoms and give decision-support suggestions
    - Classify medical images using CNN models
    - Summarize reports into simple language
    - Extract medicines from prescriptions using OCR
    - Estimate disease risk from basic patient data
    - Accept voice input and provide spoken guidance
    - Answer health questions using RAG-based retrieval
    """
)

show_disclaimer()