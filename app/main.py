import streamlit as st

st.set_page_config(
    page_title="AI Medical Decision Support",
    page_icon="🩺",
    layout="wide",
)

st.title("AI-Powered Medical Decision Support Assistant")
st.subheader("Foundation setup complete")
st.write(
    "This project will include symptom analysis, medical image classification, "
    "report summarization, prescription OCR, disease risk prediction, voice assistant, "
    "and an AI doctor chatbot."
)

st.info(
    "This is a decision-support system for educational use only. "
    "It does not replace a licensed medical professional."
)