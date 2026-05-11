import streamlit as st

from utils.ui import load_css, page_header
from services.pdf_report_service import extract_text_from_pdf, summarize_pdf_text
from services.db import insert_data


def render():
    load_css()

    page_header(
        "📄 Medical Report Summarizer",
        "Upload a PDF or paste report text for LLM-based summary"
    )

    user_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name",
        key="report_patient_name_input"
    )

    source_type = st.radio(
        "Choose input type",
        ["Paste Text", "Upload PDF"],
        horizontal=True,
        key="report_source_type"
    )

    report_text = ""

    if source_type == "Paste Text":
        report_text = st.text_area(
            "Paste Medical Report Text",
            height=250,
            placeholder="Paste blood report, discharge summary, lab report, etc.",
            key="report_text_input"
        )

    else:
        uploaded_pdf = st.file_uploader(
            "Upload Medical Report PDF",
            type=["pdf"],
            key="report_pdf_upload"
        )

        if uploaded_pdf is not None:
            with st.spinner("Extracting PDF text..."):
                report_text = extract_text_from_pdf(uploaded_pdf)

            st.session_state["latest_pdf_context"] = report_text

            with st.expander("Preview extracted text"):
                st.text_area(
                    "Extracted PDF Text",
                    value=report_text,
                    height=250,
                    key="report_pdf_preview"
                )

    if st.button("Summarize Report", key="summarize_report_button"):
        if not report_text.strip():
            st.error("Please enter or upload a report first.")
            return

        with st.spinner("Generating report summary..."):
            result = summarize_pdf_text(report_text)

        st.subheader("Summary")
        st.markdown(result["summary_markdown"])

        st.subheader("Highlights")
        if result["highlights"]:
            for item in result["highlights"]:
                st.markdown(f"- {item}")
        else:
            st.write("No highlights found.")

        payload = {
            "user_name": user_name,
            "original_text": report_text,
            "summary": result["summary_markdown"],
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