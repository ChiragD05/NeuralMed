from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Paths
# -----------------------------
KNOWLEDGE_PATH = Path("rag/medical_knowledge.txt")
VECTORSTORE_PATH = "rag/vectorstore"

# -----------------------------
# Load text
# -----------------------------
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("Loaded knowledge base")

# -----------------------------
# Split text
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.create_documents([text])

print(f"Created {len(documents)} chunks")

# -----------------------------
# Create embeddings
# -----------------------------
embeddings = OpenAIEmbeddings()

print("Generating embeddings...")

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# -----------------------------
# Save vectorstore
# -----------------------------
vectorstore.save_local(VECTORSTORE_PATH)

print(f"Saved vectorstore to {VECTORSTORE_PATH}")