import pandas as pd
import streamlit as st
from utils.ui import load_css, page_header
from services.prescription_ocr_service import extract_text_from_image, parse_medicines
from services.db import insert_data

def render():
    load_css()

    page_header(
        "💊 Prescription OCR",
        "Upload a prescription image and extract medicine details"
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="prescription_patient_name_input"
    )

    uploaded_file = st.file_uploader(
        "Upload Prescription Image",
        type=["png", "jpg", "jpeg"],
        key="prescription_upload_input"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)

        if st.button("Extract Prescription Text", key="extract_prescription_button"):
            with st.spinner("Reading prescription..."):
                text, _ = extract_text_from_image(uploaded_file)
                medicines = parse_medicines(text)

            st.subheader("Extracted Text")
            if text.strip():
                st.text_area("OCR Output", value=text, height=250, key="prescription_ocr_output")
            else:
                st.warning("No readable text detected.")

            st.subheader("Parsed Medicines")
            if medicines:
                df = pd.DataFrame(medicines)
                st.dataframe(
                    df[["medicine_name", "dosage", "frequency", "duration"]],
                    use_container_width=True
                )
            else:
                st.info("No medicines could be confidently parsed.")

            try:
                insert_data("prescriptions", {
                    "user_name": user_name,
                    "file_name": uploaded_file.name,
                    "file_url": "",
                    "extracted_text": text,
                    "medicines": medicines,
                })
                st.success("Prescription OCR saved successfully")
            except Exception as e:
                st.warning(f"Database save failed: {str(e)}")

    st.caption("This tool is for educational decision support only and does not replace a pharmacist or doctor.")