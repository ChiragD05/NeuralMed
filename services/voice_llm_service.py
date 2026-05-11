from dotenv import load_dotenv
load_dotenv()

import uuid
from pathlib import Path

from openai import OpenAI
from services.rag_chatbot_service import ask_medical_chatbot

client = OpenAI()

VOICE_AUDIO_DIR = Path("uploads/voice")
VOICE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_audio_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text.strip()


def get_medical_answer(query: str, chat_history=None) -> dict:
    return ask_medical_chatbot(query, chat_history or [])


def text_to_speech(text: str) -> str:
    out_path = VOICE_AUDIO_DIR / f"voice_reply_{uuid.uuid4().hex}.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    response.stream_to_file(str(out_path))
    return str(out_path)