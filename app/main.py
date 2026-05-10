import sys
from pathlib import Path
import importlib

import streamlit as st
from streamlit_option_menu import option_menu

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.ui import load_css, show_disclaimer
from utils.config import APP_NAME, APP_TAGLINE
from services.auth_service import (
    sign_up,
    sign_in,
    sign_out,
    restore_user_from_session,
    response_to_user_dict,
)
from services.user_context import set_auth_user, clear_auth_user, get_auth_user

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

if "auth_user" not in st.session_state:
    restored = restore_user_from_session()
    if restored:
        set_auth_user(restored)

def render_auth_gate():
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0; font-size:2.4rem;">NeuralMed AI</h1>
            <p style="margin-top:10px; font-size:1.05rem;">
                Sign in to view your personal medical AI dashboard and history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_btn"):
            try:
                resp = sign_in(email, password)
                user_dict = response_to_user_dict(resp)
                if user_dict:
                    set_auth_user(user_dict)
                    st.success("Logged in successfully.")
                    st.rerun()
                else:
                    st.error("Login worked, but no user data was returned.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_signup:
        full_name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account", key="signup_btn"):
            try:
                resp = sign_up(email, password, full_name)
                user_dict = response_to_user_dict(resp)

                if user_dict and getattr(resp, "session", None):
                    set_auth_user(user_dict)
                    st.success("Account created and signed in.")
                    st.rerun()
                else:
                    st.success(
                        "Account created. Please check your email and confirm your account before logging in."
                    )
            except Exception as e:
                st.error(f"Sign up failed: {e}")

    st.caption(
        "Supabase password auth is enabled by default on hosted projects, and sign-up may require email confirmation first."
    )

def render_dashboard():
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
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Symptom Analysis</div>
            <div class="metric-value">NLP</div>
            <div class="metric-subtitle">Triage-style guidance</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Medical Imaging</div>
            <div class="metric-value">CNN</div>
            <div class="metric-subtitle">Image classification</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Medical Reports</div>
            <div class="metric-value">NLP + LLM</div>
            <div class="metric-subtitle">Summaries in plain English</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">AI Chatbot</div>
            <div class="metric-value">RAG</div>
            <div class="metric-subtitle">Grounded medical answers</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("What this assistant can do")
    st.markdown("""
    - Understand symptoms and give decision-support suggestions
    - Classify medical images using CNN models
    - Summarize reports into simple language
    - Extract medicines from prescriptions using OCR
    - Estimate disease risk from basic patient data
    - Accept voice input and provide spoken guidance
    - Answer health questions using RAG-based retrieval
    """)

    show_disclaimer()

def render_placeholder(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0; font-size:2.1rem;">{title}</h1>
            <p style="margin-top:10px; font-size:1.02rem; opacity:0.95;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.info("This module is connected in the app shell and will be built in the next step.")

if not get_auth_user():
    render_auth_gate()
    st.stop()

with st.sidebar:
    user = get_auth_user()
    st.markdown("## 🩺 NeuralMed AI")
    st.caption(f"Logged in as: {user.get('email', 'User')}")
    if st.button("Logout", key="logout_btn"):
        try:
            sign_out()
        except Exception:
            pass
        clear_auth_user()
        st.rerun()

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Dashboard",
            "My History",
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
            "clock-history",
            "activity",
            "image",
            "file-earmark-text",
            "camera",
            "heart-pulse",
            "mic",
            "chat-dots",
        ],
        default_index=0,
        key="main_nav_v3",
    )

st.title(APP_TAGLINE)
st.write("")

if selected == "Dashboard":
    render_dashboard()

elif selected == "My History":
    history_page = importlib.import_module("features.history_dashboard")
    history_page.render()

elif selected == "Symptom Analysis":
    symptom_page = importlib.import_module("features.symptom_analysis")
    symptom_page.render()

elif selected == "Medical Imaging":
    imaging_page = importlib.import_module("features.medical_imaging")
    imaging_page.render()

elif selected == "Report Summarizer":
    report_page = importlib.import_module("features.report_summarizer")
    report_page.render()

elif selected == "Prescription OCR":
    prescription_page = importlib.import_module("features.prescription_ocr")
    prescription_page.render()

elif selected == "Risk Prediction":
    st.info("Module under development")

elif selected == "Voice Assistant":
    voice_page = importlib.import_module("features.voice_assistant")
    voice_page.render()

elif selected == "AI Chatbot":
    chatbot_page = importlib.import_module("features.ai_chatbot")
    chatbot_page.render()
