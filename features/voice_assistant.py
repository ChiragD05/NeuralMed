import tempfile
from pathlib import Path

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.ui import load_css, page_header
from services.voice_llm_service import (
    transcribe_audio_file,
    get_medical_answer,
    text_to_speech,
)

from services.db import insert_data


def render():

    load_css()

    page_header(
        "🎤 AI Voice Assistant",
        "Live microphone + OpenAI + RAG"
    )

    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []

    st.write("### Speak using your microphone")

    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    st.divider()

    typed_question = st.text_input(
        "Or type your question",
        placeholder="Example: What are symptoms of pneumonia?",
        key="voice_typed_question"
    )

    transcript = ""

    # -----------------------------
    # MICROPHONE INPUT
    # -----------------------------
    if audio_data:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(audio_data["bytes"])

            temp_audio_path = tmp_file.name

        st.audio(audio_data["bytes"])

        with st.spinner(
            "Transcribing audio..."
        ):

            transcript = transcribe_audio_file(
                temp_audio_path
            )

    # -----------------------------
    # TYPED INPUT
    # -----------------------------
    elif typed_question.strip():

        transcript = typed_question.strip()

    # -----------------------------
    # PROCESS QUERY
    # -----------------------------
    if transcript:

        st.subheader(
            "Recognized Speech"
        )

        st.info(transcript)

        st.session_state.voice_history.append({
            "role": "user",
            "content": transcript
        })

        with st.spinner(
            "Generating medical response..."
        ):

            result = get_medical_answer(
                transcript,
                st.session_state.voice_history
            )

        answer = result["answer"]

        st.session_state.voice_history.append({
            "role": "assistant",
            "content": answer
        })

        st.subheader(
            "AI Response"
        )

        st.write(answer)

        if result["sources"]:

            with st.expander(
                "Knowledge Sources"
            ):

                for src in result["sources"]:

                    st.markdown(f"- {src}")

        # -----------------------------
        # TEXT TO SPEECH
        # -----------------------------
        with st.spinner(
            "Generating voice reply..."
        ):

            reply_audio_path = text_to_speech(
                answer
            )

        st.subheader(
            "Voice Reply"
        )

        with open(
            reply_audio_path,
            "rb"
        ) as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3"
            )

        # -----------------------------
        # SAVE TO DB
        # -----------------------------
        payload = {
            "transcript": transcript,
            "assistant_reply": answer,
        }

        try:

            insert_data(
                "voice_sessions",
                payload
            )

        except Exception as e:

            st.warning(
                f"Database save failed: {str(e)}"
            )

    st.caption(
        "AI-generated voice assistance "
        "for educational decision support only."
    )