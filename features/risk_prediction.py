import streamlit as st

from utils.ui import load_css, page_header
from services.risk_model_service import predict_risk
from services.db import insert_data

def render():
    load_css()

    page_header(
        "❤️ Risk Prediction",
        "Real ML-based cardiovascular risk estimation"
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="risk_patient_name"
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, key="risk_age")
        male = st.selectbox("Sex", ["Female", "Male"], key="risk_sex")
        currentSmoker = st.selectbox("Current Smoker", ["No", "Yes"], key="risk_smoker")
        cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=0, key="risk_cigs")
        BMI = st.number_input("BMI", min_value=10.0, max_value=80.0, value=24.0, step=0.1, key="risk_bmi")
        heartRate = st.number_input("Heart Rate", min_value=30, max_value=220, value=75, key="risk_hr")
        glucose = st.number_input("Glucose", min_value=30, max_value=400, value=80, key="risk_glucose")

    with col2:
        BPMeds = st.selectbox("On Blood Pressure Medication", ["No", "Yes"], key="risk_bpmeds")
        prevalentStroke = st.selectbox("Previous Stroke", ["No", "Yes"], key="risk_stroke")
        prevalentHyp = st.selectbox("Hypertension", ["No", "Yes"], key="risk_hyp")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="risk_diabetes")
        totChol = st.number_input("Total Cholesterol", min_value=50, max_value=600, value=180, key="risk_chol")
        sysBP = st.number_input("Systolic BP", min_value=50, max_value=300, value=120, key="risk_sysbp")
        diaBP = st.number_input("Diastolic BP", min_value=30, max_value=200, value=80, key="risk_diabp")

    if st.button("Predict Risk", key="predict_risk_button"):
        input_data = {
            "male": 1 if male == "Male" else 0,
            "age": age,
            "currentSmoker": 1 if currentSmoker == "Yes" else 0,
            "cigsPerDay": cigsPerDay,
            "BPMeds": 1 if BPMeds == "Yes" else 0,
            "prevalentStroke": 1 if prevalentStroke == "Yes" else 0,
            "prevalentHyp": 1 if prevalentHyp == "Yes" else 0,
            "diabetes": 1 if diabetes == "Yes" else 0,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose,
        }

        try:
            result = predict_risk(input_data)

            st.subheader("Prediction Result")
            st.metric("Risk Label", result["risk_label"])
            st.metric("Risk Probability", f"{result['risk_probability']}%")

            if result["risk_probability"] >= 50:
                st.warning("Higher predicted cardiovascular risk. Please treat this as decision support only.")
            else:
                st.success("Lower predicted cardiovascular risk.")

            payload = {
                "user_name": user_name,
                "patient_data": input_data,
                "prediction": result,
            }

            try:
                insert_data("risk_predictions", payload)
                st.success("Risk prediction saved successfully")
            except Exception as e:
                st.warning(f"Database save failed: {str(e)}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    st.caption(
        "This tool is for educational decision support only and does not replace a licensed clinician."
    )