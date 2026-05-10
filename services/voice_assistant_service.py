import os
import tempfile
from pathlib import Path

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

# Explicit ffmpeg paths for Apple Silicon Macs
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe   = "/opt/homebrew/bin/ffprobe"


def convert_to_pcm_wav(input_path: str) -> str:
    """
    Convert any audio file to PCM WAV so SpeechRecognition can read it.
    Requires ffmpeg installed for non-wav formats.
    """
    input_path = str(input_path)
    output_path = str(Path(tempfile.gettempdir()) / "medical_ai_input_pcm.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)        # mono
    audio = audio.set_frame_rate(16000)  # speech-friendly
    audio.export(output_path, format="wav")

    return output_path


def speech_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()

    # Convert first to a clean PCM WAV
    wav_path = convert_to_pcm_wav(audio_path)

    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Speech recognition service unavailable."


def text_to_speech(text: str) -> str:
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "medical_ai_response.mp3")

    tts = gTTS(text=text, lang="en")
    tts.save(output_path)

    return output_path