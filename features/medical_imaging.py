import streamlit as st

from utils.ui import load_css, page_header
from utils.charts import imaging_prediction_chart

from services.medical_imaging_service import (
    analyze_medical_image
)

from services.db import insert_data


def render():

    load_css()

    page_header(
        "🩻 Medical Imaging AI",
        "Real CNN-based medical image analysis"
    )

    uploaded_image = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"],
        key="medical_image_upload"
    )

    if uploaded_image is not None:
        try:
           uploaded_image.seek(0)
        except Exception:
            pass

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Analyze Medical Image",
            key="analyze_medical_image_button"
        ):

            with st.spinner(
                "Running CNN inference..."
            ):

                result = analyze_medical_image(
                    uploaded_image
                )

            st.success(
                "AI analysis complete"
            )

            st.subheader(
                "Prediction"
            )

            st.metric(
                "Top Prediction",
                result["top_prediction"]
            )

            st.metric(
                "Confidence",
                f"{result['confidence']}%"
            )

            st.info(
                result["description"]
            )

            st.subheader(
                "Prediction Breakdown"
            )

            chart = imaging_prediction_chart(
                result["all_predictions"]
            )

            if chart:
                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

            payload = {
                "prediction":
                    result["top_prediction"],

                "confidence":
                    result["confidence"],
            }

            try:

                insert_data(
                    "medical_images",
                    payload
                )

            except Exception as e:

                st.warning(
                    f"Database save failed: {str(e)}"
                )

    st.caption(
        "AI-generated medical imaging analysis "
        "for educational decision support only."
    )