import streamlit as st

from utils.ui import load_css, page_header
from services.pdf_report_service import extract_text_from_pdf
from services.rag_chatbot_service import ask_medical_chatbot
from services.db import insert_data


def render():
    load_css()

    page_header(
        "🤖 AI Medical Chatbot",
        "LangChain + OpenAI + FAISS + DuckDuckGo + PDF context"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_pdf_context" not in st.session_state:
        st.session_state.chat_pdf_context = ""

    uploaded_pdf = st.file_uploader(
        "Optional: upload a medical report PDF for context",
        type=["pdf"],
        key="chat_pdf_upload"
    )

    if uploaded_pdf is not None:
        with st.spinner("Extracting PDF text..."):
            pdf_text = extract_text_from_pdf(uploaded_pdf)

        st.session_state.chat_pdf_context = pdf_text

        st.success("PDF context loaded for this chat session.")

        with st.expander("Preview extracted PDF text"):
            st.text_area(
                "Extracted text",
                value=pdf_text,
                height=220,
                key="chat_pdf_preview"
            )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear PDF context", key="clear_pdf_context_button"):
            st.session_state.chat_pdf_context = ""
            st.success("PDF context cleared.")
    with col2:
        st.caption("Chat memory is kept in this session.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a medical question...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            result = ask_medical_chatbot(
                query,
                st.session_state.chat_history,
                pdf_context=st.session_state.chat_pdf_context,
            )

        answer = result["answer"]
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            if result["sources"]:
                with st.expander("Knowledge Sources"):
                    for src in result["sources"]:
                        st.markdown(f"- {src}")

        try:
            insert_data(
                "chat_messages",
                {
                    "query": query,
                    "response": answer,
                    "retrieved_context": result["sources"],
                },
            )
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption(
        "AI-generated medical responses for educational decision support only."
    )