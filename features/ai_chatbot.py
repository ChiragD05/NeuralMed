import streamlit as st

from utils.ui import load_css, page_header
from services.rag_chatbot_service import ask_medical_chatbot
from services.db import insert_data


def render():
    load_css()

    page_header(
        "🤖 AI Medical Chatbot",
        "LangChain + OpenAI + FAISS + DuckDuckGo"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a medical question...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            result = ask_medical_chatbot(query, st.session_state.chat_history)

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
                },
            )
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption(
        "AI-generated medical responses for educational decision support only."
    )