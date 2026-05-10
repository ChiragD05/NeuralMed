import streamlit as st
from utils.ui import load_css, page_header
from services.symptom_analysis_service import analyze_symptoms
from services.db import insert_data
from utils.charts import prediction_chart

def render():
    load_css()

    page_header(
        "🩺 Symptom Analysis",
        "AI-powered symptom understanding and triage"
    )

    st.markdown(
        """
        Enter symptoms in natural language.

        Example:
        - fever, dry cough, fatigue
        - chest pain and shortness of breath
        - headache with nausea
        """
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="symptom_patient_name_input_v2"
    )

    symptoms = st.text_area(
        "Describe Symptoms",
        height=180,
        placeholder="Describe symptoms in detail...",
        key="symptom_description_input_v2"
    )

    analyze_btn = st.button(
        "Analyze Symptoms",
        key="analyze_symptoms_button_v2"
    )

    if analyze_btn:
        if not symptoms.strip():
            st.error("Please enter symptoms")
            return

        with st.spinner("Analyzing symptoms..."):
            results = analyze_symptoms(symptoms)

        severity = results["severity"]

        if severity == "High":
            st.error(f"Severity Level: {severity}")
        elif severity == "Moderate to High":
            st.warning(f"Severity Level: {severity}")
        else:
            st.success(f"Severity Level: {severity}")

        if results["emergency"]:
            st.error("⚠ Emergency indicators detected. Immediate medical attention recommended.")

        st.subheader("Top Predicted Conditions")
        predictions = results["predictions"]

        if predictions:
            for prediction in predictions:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">{prediction["condition"]}</div>
                        <div class="metric-value">{prediction["confidence"]}%</div>
                        <div class="metric-subtitle">
                            Matched symptoms: {", ".join(prediction["matched_symptoms"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")

            chart = prediction_chart(predictions)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("No matching condition patterns found.")

        st.subheader("Recommendation")
        st.info(results["recommendation"])

        try:
            insert_data("symptom_logs", {
                "user_name": user_name,
                "symptoms": symptoms,
                "analysis": results,
            })
            st.success("Analysis saved successfully")
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.divider()
    st.caption(
        "This system provides AI-assisted decision support only and does not replace licensed medical professionals."
    )