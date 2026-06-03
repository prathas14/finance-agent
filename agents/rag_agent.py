from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

INDEX_PATH = str(Path(__file__).parent.parent / "knowledge_base" / ".faiss")

DISCLAIMER = "\n\n⚠️ *This is for educational purposes only and not financial advice.*"

_vectorstore = None
_chain = None


def _get_chain():
    global _vectorstore, _chain
    if _chain:
        return _chain

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    _vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        """You are a friendly financial literacy educator. Use the context below to answer the question.
If the answer is not in the context, say you don't have enough information but provide general guidance.
Keep answers clear, concise, and beginner-friendly.

Context:
{context}

Question: {question}

Answer:"""
    )

    _chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return _chain


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def answer(question: str) -> str:
    try:
        chain = _get_chain()
        result = chain.invoke(question)
        return result.content + DISCLAIMER
    except Exception as e:
        return f"RAG agent error: {e}{DISCLAIMER}"
