import re
from collections import Counter
from pypdf import PdfReader

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "will", "with", "patient", "report", "date",
    "normal", "mild", "moderate", "severe"
}

KEY_MEDICAL_TERMS = [
    "fever", "cough", "pain", "infection", "normal", "abnormal", "high",
    "low", "positive", "negative", "diabetes", "hypertension", "bp",
    "glucose", "hemoglobin", "platelet", "cholesterol", "creatinine",
    "sugar", "urine", "blood", "xray", "x-ray", "pneumonia"
]

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts).strip()

def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]

def tokenize(text: str):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def sentence_score(sentence: str, word_freq: Counter):
    words = tokenize(sentence)
    if not words:
        return 0
    score = sum(word_freq[w] for w in words)
    return score / len(words)

def generate_summary(text: str, max_sentences: int = 4):
    text = text.strip()
    if not text:
        return {
            "summary": "",
            "highlights": [],
            "key_terms": [],
        }

    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        highlights = [s for s in sentences[:max_sentences]]
        key_terms = find_key_terms(text)
        return {
            "summary": " ".join(sentences),
            "highlights": highlights,
            "key_terms": key_terms,
        }

    words = tokenize(text)
    word_freq = Counter(words)

    if not word_freq:
        key_terms = find_key_terms(text)
        return {
            "summary": " ".join(sentences[:max_sentences]),
            "highlights": sentences[:max_sentences],
            "key_terms": key_terms,
        }

    max_freq = max(word_freq.values())
    for word in list(word_freq.keys()):
        word_freq[word] = word_freq[word] / max_freq

    ranked = []
    for i, sentence in enumerate(sentences):
        score = sentence_score(sentence, word_freq)
        ranked.append((i, score, sentence))

    top_sentences = sorted(
        ranked,
        key=lambda x: x[1],
        reverse=True
    )[:max_sentences]

    top_sentences = sorted(top_sentences, key=lambda x: x[0])

    summary = " ".join([s[2] for s in top_sentences])
    highlights = [s[2] for s in top_sentences]
    key_terms = find_key_terms(text)

    return {
        "summary": summary,
        "highlights": highlights,
        "key_terms": key_terms,
    }

def find_key_terms(text: str):
    lower = text.lower()
    found = []

    for term in KEY_MEDICAL_TERMS:
        if term in lower and term not in found:
            found.append(term)

    return found[:10]

def simplify_report(text: str):
    result = generate_summary(text, max_sentences=4)

    if result["key_terms"]:
        key_terms_text = ", ".join(result["key_terms"])
    else:
        key_terms_text = "No strong medical keywords detected."

    return {
        "summary": result["summary"],
        "highlights": result["highlights"],
        "key_terms": result["key_terms"],
        "key_terms_text": key_terms_text,
    }