import uuid
from datetime import datetime

import streamlit as st

from utils.ui import load_css, page_header
from services.rag_chatbot_service import ask_medical_chatbot
from services.db import insert_data, fetch_all_rows
from services.user_context import get_auth_user


def _parse_dt(value):
    try:
        if not value:
            return datetime.min
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _get_user_email():
    user = get_auth_user()
    return user.get("email") if user else None


def _load_sessions():
    try:
        result = fetch_all_rows("chat_messages")
        rows = result.data if hasattr(result, "data") else []
    except Exception:
        rows = []

    user_email = _get_user_email()
    if user_email:
        user_rows = [r for r in rows if r.get("app_user_email") == user_email]
        if user_rows:
            rows = user_rows

    sessions = {}

    for row in rows:
        sid = row.get("chat_session_id") or f"legacy-{row.get('id')}"
        title = row.get("chat_session_title") or (row.get("query") or "Chat")[:50]
        created_at = row.get("created_at")

        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "title": title,
                "created_at": created_at,
                "rows": [],
            }

        sessions[sid]["rows"].append(row)

        if _parse_dt(created_at) > _parse_dt(sessions[sid]["created_at"]):
            sessions[sid]["created_at"] = created_at
            sessions[sid]["title"] = title

    session_list = list(sessions.values())
    session_list.sort(key=lambda x: _parse_dt(x["created_at"]), reverse=True)
    return session_list


def _rows_to_messages(rows):
    msgs = []
    rows = sorted(rows, key=lambda r: _parse_dt(r.get("created_at")))
    for row in rows:
        q = (row.get("query") or "").strip()
        a = (row.get("response") or "").strip()
        if q:
            msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs


def _reset_chat():
    st.session_state.current_chat_session_id = None
    st.session_state.current_chat_session_title = None
    st.session_state.chat_history = []


def _load_session(session):
    st.session_state.current_chat_session_id = session["session_id"]
    st.session_state.current_chat_session_title = session["title"]
    st.session_state.chat_history = _rows_to_messages(session["rows"])


def render():
    load_css()

    page_header(
        "🤖 AI Medical Chatbot",
        "LangChain + OpenAI + FAISS + DuckDuckGo"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "current_chat_session_id" not in st.session_state:
        st.session_state.current_chat_session_id = None

    if "current_chat_session_title" not in st.session_state:
        st.session_state.current_chat_session_title = None

    sessions = _load_sessions()

    with st.sidebar:
        st.markdown("### 🕘 Previous Chats")

        if st.button("➕ New Chat", key="new_chat_button"):
            _reset_chat()
            st.rerun()

        if sessions:
            for session in sessions[:15]:
                label = session["title"] or "Chat"
                if st.button(label[:45], key=f"load_{session['session_id']}"):
                    _load_session(session)
                    st.rerun()
        else:
            st.caption("No saved chats yet.")

    # Show current chat only
    if st.session_state.chat_history:
        st.markdown(f"**Current chat:** {st.session_state.current_chat_session_title or 'Conversation'}")
        st.write("")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("Start a new chat or click a previous chat from the sidebar.")

    query = st.chat_input("Ask a medical question...")

    if query:
        if not st.session_state.current_chat_session_id:
            st.session_state.current_chat_session_id = uuid.uuid4().hex
            st.session_state.current_chat_session_title = query[:50]

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
                    "chat_session_id": st.session_state.current_chat_session_id,
                    "chat_session_title": st.session_state.current_chat_session_title,
                    "query": query,
                    "response": answer,
                    "retrieved_context": result["sources"],
                },
            )
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")

    st.caption("AI-generated medical responses for educational decision support only.")