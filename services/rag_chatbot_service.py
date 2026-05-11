from dotenv import load_dotenv
load_dotenv()

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage

VECTORSTORE_PATH = "rag/vectorstore"

SYSTEM_PROMPT = """
You are NeuralMed AI, a warm and helpful medical decision-support assistant.

Style rules:
- Sound natural and conversational, not robotic.
- Answer in 1 to 3 short paragraphs.
- Use simple language a patient can understand.
- Do not repeat retrieved text verbatim.
- Do not dump bullet lists unless the user explicitly asks for them.
- If the question is a greeting or casual message, reply naturally and briefly.
- If context is weak, say so politely and suggest a licensed clinician or ask one short follow-up question.
- Never claim to replace a doctor.
"""

WEB_SEARCH = DuckDuckGoSearchResults(output_format="list")


@lru_cache(maxsize=1)
def load_components():
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-5.5",
        temperature=0.7,
    )

    return retriever, llm


def format_history(chat_history):
    if not chat_history:
        return ""

    lines = []
    for item in chat_history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role and content:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def retrieve_local_context(query: str):
    retriever, _ = load_components()
    docs = retriever.invoke(query)
    return docs


def web_search_context(query: str):
    try:
        results = WEB_SEARCH.invoke(query)
    except Exception:
        return [], ""

    if not isinstance(results, list):
        return [], ""

    sources = []
    snippets = []

    for item in results[:3]:
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        link = item.get("link", "").strip()

        if title or snippet:
            sources.append({
                "title": title,
                "snippet": snippet,
                "link": link,
            })

            snippets.append(
                f"Title: {title}\nSnippet: {snippet}\nLink: {link}"
            )

    return sources, "\n\n".join(snippets)


def needs_web_search(query: str, local_docs):
    q = query.lower()

    fresh_terms = [
        "latest", "current", "today", "recent", "new", "news",
        "guideline", "guidelines", "update", "updates", "outbreak"
    ]

    if any(term in q for term in fresh_terms):
        return True

    if len(local_docs) == 0:
        return True

    local_text = " ".join(doc.page_content for doc in local_docs).strip()
    if len(local_text) < 200:
        return True

    return False


def ask_medical_chatbot(query: str, chat_history=None):
    query_clean = query.strip()

    if not query_clean:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    greetings = {"hi", "hello", "hey", "good morning", "good evening"}
    if query_clean.lower() in greetings:
        return {
            "answer": (
                "Hello! I am NeuralMed AI. "
                "Ask me about symptoms, reports, medicines, or general health guidance, "
                "and I will help in simple language."
            ),
            "sources": [],
        }

    retriever, llm = load_components()
    local_docs = retriever.invoke(query_clean)

    local_context = "\n\n".join(
        doc.page_content.strip()
        for doc in local_docs[:4]
        if doc.page_content.strip()
    )

    web_sources = []
    web_context = ""

    if needs_web_search(query_clean, local_docs):
        web_sources, web_context = web_search_context(query_clean)

    history_text = format_history(chat_history)

    prompt = f"""
User question:
{query_clean}

Relevant local medical context:
{local_context if local_context else "No strong local context retrieved."}

Web search context:
{web_context if web_context else "No web search context used."}

Conversation history:
{history_text if history_text else "No prior conversation."}

Now write the best possible answer.
Keep it warm, natural, and medically cautious.
Do not sound rigid.
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    ).content.strip()

    sources = []
    for doc in local_docs[:3]:
        text = doc.page_content.strip()
        if text and text not in sources:
            sources.append(text)

    for item in web_sources:
        snippet = item.get("snippet", "").strip()
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        source_text = f"{title} — {snippet} — {link}".strip(" —")
        if source_text and source_text not in sources:
            sources.append(source_text)

    return {
        "answer": response,
        "sources": sources[:5],
    }