import streamlit as st

from utils.ui import load_css, page_header
from services.medical_imaging_service import analyze_medical_image
from utils.charts import imaging_prediction_chart
from services.db import insert_data

def render():

    load_css()

    page_header(
        "🩻 Medical Image Classification",
        "CNN-powered medical image analysis"
    )

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="imaging_patient_name"
    )

    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["png", "jpg", "jpeg"],
        key="medical_image_upload"
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Medical Image",
            use_container_width=True
        )

        analyze_btn = st.button(
            "Analyze Medical Image",
            key="analyze_medical_image_button"
        )

        if analyze_btn:

            with st.spinner("Running CNN inference..."):

                results = analyze_medical_image(uploaded_file)

            st.subheader("Prediction Result")

            confidence = results["confidence"]

            if confidence > 75:
                st.error(
                    f"Primary Prediction: {results['top_prediction']} ({confidence}%)"
                )

            elif confidence > 50:
                st.warning(
                    f"Primary Prediction: {results['top_prediction']} ({confidence}%)"
                )

            else:
                st.success(
                    f"Primary Prediction: {results['top_prediction']} ({confidence}%)"
                )

            st.info(results["description"])

            st.subheader("Prediction Confidence Scores")

            chart = imaging_prediction_chart(
                results["all_predictions"]
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            st.subheader("Detailed Probabilities")

            for prediction in results["all_predictions"]:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">
                            {prediction["class"]}
                        </div>

                        <div class="metric-value">
                            {prediction["confidence"]}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

            payload = {
                "user_name": patient_name,
                "file_name": uploaded_file.name,
                "file_url": "",
                "prediction": results,
            }

            try:

                insert_data(
                    "medical_images",
                    payload
                )

                st.success(
                    "Medical imaging result saved successfully"
                )

            except Exception as e:

                st.warning(
                    f"Database save failed: {str(e)}"
                )

    st.caption(
        "This tool provides AI-assisted decision support only "
        "and does not replace radiologists or licensed medical professionals."
    )