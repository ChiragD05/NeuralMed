from dotenv import load_dotenv
load_dotenv()

from functools import lru_cache
from pathlib import Path

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "rag/vectorstore"

SYSTEM_STYLE = """
You are NeuralMed AI, a warm, clear medical decision-support assistant.

Style rules:
- Sound natural and conversational, not robotic.
- Answer in 1 to 3 short paragraphs.
- Use simple language a patient can understand.
- Do not repeat retrieved text verbatim.
- Do not dump bullet lists unless the user explicitly asks for them.
- If the question is a greeting or casual message, reply naturally and briefly.
- If the retrieved context is weak, say so politely and ask one short follow-up question or suggest consulting a licensed clinician.
- Never claim to replace a doctor.
"""

ANSWER_PROMPT = PromptTemplate.from_template(
    SYSTEM_STYLE
    + """

Retrieved medical context:
{context}

Conversation history:
{chat_history}

User question:
{question}

Write the best possible answer now:
"""
)

@lru_cache(maxsize=1)
def _load_chain():
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

    memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True,
)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
    )

    return chain

def ask_medical_chatbot(query: str):
    query_clean = query.strip()

    if not query_clean:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # Friendly direct response for greetings
    greetings = {"hi", "hello", "hey", "good morning", "good evening"}
    if query_clean.lower() in greetings:
        return {
            "answer": "Hello! I am NeuralMed AI. Ask me about symptoms, reports, medicines, or general health guidance, and I will help in simple language.",
            "sources": [],
        }

    chain = _load_chain()
    result = chain.invoke({"question": query_clean})

    answer = result.get("answer", "").strip()
    source_docs = result.get("source_documents", [])

    sources = []
    for doc in source_docs:
        text = doc.page_content.strip()
        if text and text not in sources:
            sources.append(text)

    return {
        "answer": answer,
        "sources": sources[:3],
    }