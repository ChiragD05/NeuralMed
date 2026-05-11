from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model=MODEL_NAME)

def chunk_text(text: str, max_words: int = 350):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks

def summarize_medical_report(text: str):
    text = text.strip()

    if not text:
        return {
            "summary": "",
            "highlights": [],
        }

    summarizer = get_summarizer()

    chunks = chunk_text(text, max_words=350)
    chunk_summaries = []

    for chunk in chunks:
        try:
            result = summarizer(
                chunk,
                max_length=120,
                min_length=35,
                do_sample=False
            )
            chunk_summaries.append(result[0]["summary_text"])
        except Exception:
            # fallback to chunk text if summarization fails
            chunk_summaries.append(chunk[:250])

    final_summary = " ".join(chunk_summaries)

    highlights = []
    for sentence in final_summary.split("."):
        sentence = sentence.strip()
        if sentence:
            highlights.append(sentence)

    return {
        "summary": final_summary,
        "highlights": highlights[:5],
    }