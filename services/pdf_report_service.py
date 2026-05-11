from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import fitz  # PyMuPDF

client = OpenAI()


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)

    return "\n\n".join(pages).strip()


def chunk_text(text: str, max_chars: int = 3500):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def _call_llm(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return response.output_text.strip()


def summarize_pdf_text(text: str) -> dict:
    text = text.strip()

    if not text:
        return {
            "summary_markdown": "",
            "highlights": [],
            "extracted_text": "",
        }

    chunks = chunk_text(text, max_chars=3500)
    chunk_notes = []

    for idx, chunk in enumerate(chunks[:6]):
        prompt = f"""
You are a medical report assistant.

Summarize this medical PDF chunk in simple, patient-friendly English.
Focus on:
- main findings
- abnormal values
- impression/diagnosis
- medicines or recommendations
- anything important for follow-up

Do not invent facts.

Chunk {idx + 1}:
{chunk}
"""
        chunk_notes.append(_call_llm(prompt))

    combined_notes = "\n\n".join(
        f"Chunk summary {i + 1}:\n{note}"
        for i, note in enumerate(chunk_notes)
    )

    final_prompt = f"""
You are a medical report assistant.

Turn the chunk summaries below into a clean markdown report with these sections:

## Simple Summary
## Key Findings
## Possible Follow-up
## Safety Note

Rules:
- Keep it clear and not too long.
- Use plain language.
- Do not invent facts.
- If something is uncertain, say so.

Chunk summaries:
{combined_notes}
"""
    summary_markdown = _call_llm(final_prompt)

    highlights = []
    for line in summary_markdown.splitlines():
        clean = line.strip()
        if clean.startswith("-") or clean.startswith("•"):
            highlights.append(clean.lstrip("-•").strip())

    return {
        "summary_markdown": summary_markdown,
        "highlights": highlights[:6],
        "extracted_text": text,
    }