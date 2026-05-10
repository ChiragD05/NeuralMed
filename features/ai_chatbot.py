import streamlit as st

from utils.ui import load_css, page_header
from services.chatbot_service import generate_response
from services.db import insert_data


def render():
    load_css()

    page_header(
        "🤖 AI Doctor Chatbot",
        "RAG-style healthcare assistant"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a medical question...")

    if user_query:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            result = generate_response(user_query)

        response = result["response"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            if result["emergency"]:
                st.error("Emergency indicators detected.")

            st.markdown(response)

            if result["sources"]:
                with st.expander("Knowledge Sources"):
                    for src in result["sources"]:
                        st.markdown(f"- {src}")

        payload = {
            "user_name": "guest",
            "query": user_query,
            "response": response,
            "retrieved_context": result["sources"],
        }

        try:
            insert_data("chat_messages", payload)
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption(
        "This chatbot provides educational decision support only and does not replace licensed medical professionals."
    )