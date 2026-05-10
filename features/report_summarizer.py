import streamlit as st
from utils.ui import load_css, page_header
from services.report_summarizer_service import extract_text_from_pdf, simplify_report
from services.db import insert_data

def render():
    load_css()

    page_header(
        "📄 Medical Report Summarizer",
        "Upload a PDF or paste report text to get a simplified summary"
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

    elif source_type == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload PDF Report",
            type=["pdf"],
            key="report_pdf_upload"
        )

        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                report_text = extract_text_from_pdf(uploaded_pdf)

            st.subheader("Extracted Report Text")
            st.text_area(
                "PDF Text",
                value=report_text,
                height=250,
                key="report_pdf_text_output"
            )

    if st.button("Summarize Report", key="summarize_report_button"):
        if not report_text.strip():
            st.error("Please enter or upload a report first.")
            return

        with st.spinner("Generating summary..."):
            result = simplify_report(report_text)

        st.subheader("Simple Summary")
        st.info(result["summary"] if result["summary"] else "No summary could be generated.")

        st.subheader("Key Medical Terms")
        st.write(result["key_terms_text"])

        st.subheader("Important Highlights")
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