import re
from pathlib import Path

KNOWLEDGE_PATH = Path("rag/medical_knowledge.txt")

EMERGENCY_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "unconscious",
    "stroke",
    "severe bleeding",
    "heart attack",
]

def load_knowledge():
    if not KNOWLEDGE_PATH.exists():
        return []

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]

    return chunks

KNOWLEDGE_BASE = load_knowledge()

def tokenize(text):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))

def retrieve_relevant_chunks(query, top_k=3):
    query_tokens = tokenize(query)

    scored = []

    for chunk in KNOWLEDGE_BASE:
        chunk_tokens = tokenize(chunk)

        overlap = len(query_tokens.intersection(chunk_tokens))

# boost exact disease mentions
        query_lower = query.lower()

        if "pneumonia" in query_lower and "pneumonia" in chunk.lower():
         overlap += 5

        if "covid" in query_lower and "covid" in chunk.lower():
           overlap += 5

        scored.append((overlap, chunk))

    scored.sort(reverse=True)

    results = [chunk for score, chunk in scored if score > 0]

    return results[:2]

def detect_emergency(query):
    lower = query.lower()

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in lower:
            return True

    return False

def generate_response(query):
    relevant_chunks = retrieve_relevant_chunks(query)

    emergency = detect_emergency(query)

    if emergency:
        emergency_text = (
            "⚠ Your symptoms may require urgent medical attention. "
            "Please contact emergency services or seek immediate care."
        )
    else:
        emergency_text = ""

    if relevant_chunks:
        answer = "\n\n".join(relevant_chunks)
    else:
        answer = (
            "I could not find a confident medical reference in the current knowledge base. "
            "Please consult a licensed medical professional."
        )

    final_response = f"""
{emergency_text}

{answer}

This response is AI-generated for educational decision support only.
"""

    return {
        "response": final_response.strip(),
        "sources": relevant_chunks,
        "emergency": emergency,
    }