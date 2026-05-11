import streamlit as st

from utils.ui import load_css, page_header
from utils.charts import prediction_chart
from services.symptom_ml_service import predict_symptom_disease
from services.db import insert_data


def render():
    load_css()

    page_header(
        "🩺 Symptom Analysis",
        "Real ML-based symptom-to-disease prediction"
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="symptom_patient_name_input"
    )

    symptoms = st.text_area(
        "Describe Symptoms",
        height=180,
        placeholder="Example: fever, cough, fatigue, chest pain",
        key="symptom_description_input"
    )

    if st.button("Analyze Symptoms", key="analyze_symptoms_button"):
        if not symptoms.strip():
            st.error("Please enter symptoms first.")
            return

        with st.spinner("Running symptom model..."):
            result = predict_symptom_disease(symptoms)

        st.subheader("Prediction Result")
        st.metric("Top Prediction", result["top_prediction"])
        st.metric("Confidence", f"{result['confidence']}%")

        st.subheader("Top Probabilities")
        chart = prediction_chart(result["all_predictions"])
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        if result["confidence"] < 60:
            st.warning(
                "Low confidence prediction. Please treat this as decision support only."
            )

        payload = {
            "user_name": user_name,
            "symptoms": symptoms,
            "analysis": result,
        }

        try:
            insert_data("symptom_logs", payload)
            st.success("Symptom analysis saved successfully")
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption(
        "This tool provides AI-assisted decision support only and does not replace licensed medical professionals."
    )