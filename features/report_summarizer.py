import streamlit as st

from utils.ui import load_css, page_header
from services.report_ml_service import summarize_medical_report
from services.db import insert_data

def render():
    load_css()

    page_header(
        "📄 Medical Report Summarizer",
        "Transformer-based report understanding and simplification"
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="report_patient_name_input"
    )

    report_text = st.text_area(
        "Paste Medical Report Text",
        height=250,
        placeholder="Paste blood report, discharge summary, lab report, etc.",
        key="report_text_input"
    )

    if st.button("Summarize Report", key="summarize_report_button"):
        if not report_text.strip():
            st.error("Please paste a report first.")
            return

        with st.spinner("Generating transformer summary..."):
            result = summarize_medical_report(report_text)

        st.subheader("Summary")
        st.info(result["summary"] if result["summary"] else "No summary generated.")

        st.subheader("Highlights")
        if result["highlights"]:
            for item in result["highlights"]:
                st.markdown(f"- {item}")
        else:
            st.write("No highlights found.")

        payload = {
            "user_name": user_name,
            "original_text": report_text,
            "summary": result["summary"],
            "highlights": result["highlights"],
        }

        try:
            insert_data("report_summaries", payload)
            st.success("Report summary saved successfully")
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption(
        "This tool is for educational decision support only and does not replace a licensed medical professional."
    )