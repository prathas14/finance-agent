from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

INDEX_PATH = str(Path(__file__).parent.parent / "knowledge_base" / ".faiss")

DISCLAIMER = "\n\n⚠️ *This is for educational purposes only and not financial advice.*"

_vectorstore = None


def _get_retriever():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore.as_retriever(search_kwargs={"k": 4})


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _rewrite_query(question: str, prior: list, llm) -> str:
    """Rewrite a follow-up question into a standalone search query using conversation history."""
    rewrite_messages = [
        SystemMessage(content=(
            "Given the conversation history below, rewrite the user's latest question as a "
            "short, self-contained search query. Output only the rewritten query, nothing else."
        )),
        *prior,
        HumanMessage(content=question),
    ]
    return llm.invoke(rewrite_messages).content.strip()


def answer(question: str, history: list = []) -> str:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # include prior turns (exclude the current HumanMessage at the end)
        prior = history[:-1] if history else []

        # rewrite vague follow-ups into standalone queries before FAISS retrieval
        search_query = _rewrite_query(question, prior, llm) if prior else question

        retriever = _get_retriever()
        context = _format_docs(retriever.invoke(search_query))

        system = SystemMessage(content=(
            "You are a friendly financial literacy educator. Use the context below to answer the question.\n"
            "If the answer is not in the context, say you don't have enough information but provide general guidance.\n"
            "Keep answers clear, concise, and beginner-friendly.\n\n"
            f"Context:\n{context}"
        ))

        messages = [system] + prior + [HumanMessage(content=question)]

        result = llm.invoke(messages)
        return result.content + DISCLAIMER
    except Exception as e:
        return f"RAG agent error: {e}{DISCLAIMER}"
