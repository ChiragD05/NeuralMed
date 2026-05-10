import tempfile
import streamlit as st

from streamlit_mic_recorder import mic_recorder

from utils.ui import load_css, page_header
from services.voice_assistant_service import speech_to_text, text_to_speech
from services.chatbot_service import generate_response
from services.db import insert_data


def render():
    load_css()

    page_header(
        "🎤 AI Voice Assistant",
        "Speech-enabled healthcare assistant"
    )

    st.subheader("Record your voice")

    audio_data = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=False,
        use_container_width=True,
        key="voice_recorder"
    )

    if audio_data and "bytes" in audio_data:
        st.audio(audio_data["bytes"], format="audio/wav")

        if st.button("Process Voice Query", key="process_voice_button"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data["bytes"])
                temp_audio_path = tmp_file.name

            with st.spinner("Converting speech to text..."):
                query = speech_to_text(temp_audio_path)

            st.subheader("Recognized Speech")
            st.info(query)

            with st.spinner("Generating AI response..."):
                result = generate_response(query)

            response = result["response"]

            st.subheader("AI Response")
            st.markdown(response)

            if result["emergency"]:
                st.error("Emergency indicators detected.")

            with st.spinner("Generating voice response..."):
                audio_response_path = text_to_speech(response)

            st.subheader("Voice Response")
            with open(audio_response_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")

            payload = {
                "user_query": query,
                "assistant_response": response,
            }

            try:
                insert_data("voice_sessions", payload)
                st.success("Voice session saved successfully")
            except Exception as e:
                st.warning(f"Database save failed: {str(e)}")

    st.divider()
    st.caption(
        "This voice assistant provides educational decision support only and does not replace licensed medical professionals."
    )