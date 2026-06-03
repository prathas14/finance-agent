"""
Run once to build the FAISS index from articles + glossary.
Usage: python knowledge_base/ingest.py
"""
import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

ARTICLES_DIR = Path(__file__).parent / "articles"
GLOSSARY_FILE = Path(__file__).parent / "glossary.json"
INDEX_PATH = Path(__file__).parent / ".faiss"


def load_documents() -> list[dict]:
    docs = []
    for md_file in ARTICLES_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        docs.append({"content": text, "source": md_file.name})

    glossary = json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
    glossary_text = "\n".join(f"{term}: {definition}" for term, definition in glossary.items())
    docs.append({"content": glossary_text, "source": "glossary.json"})

    return docs


def build_index():
    print("Loading documents...")
    raw_docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    metadatas = []
    for doc in raw_docs:
        splits = splitter.split_text(doc["content"])
        chunks.extend(splits)
        metadatas.extend([{"source": doc["source"]}] * len(splits))

    print(f"Created {len(chunks)} chunks. Building FAISS index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
