import pytest
from unittest.mock import patch, MagicMock


@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_answer_returns_disclaimer(mock_retriever, mock_llm_cls):
    mock_retriever.return_value.invoke.return_value = []

    mock_result = MagicMock()
    mock_result.content = "An ETF is an exchange-traded fund."
    mock_llm_cls.return_value.invoke.return_value = mock_result

    from agents.rag_agent import answer
    response = answer("What is an ETF?")

    assert "ETF" in response
    assert "educational purposes" in response or "financial advice" in response


@patch("agents.rag_agent._get_retriever")
def test_answer_handles_exception(mock_retriever):
    mock_retriever.side_effect = Exception("Index not found")

    from agents.rag_agent import answer
    response = answer("What is a bond?")

    assert "error" in response.lower() or "RAG" in response


# ---------------------------------------------------------------------------
# Query rewriting — follow-up questions use history-aware FAISS search query
# ---------------------------------------------------------------------------

@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_no_rewrite_when_no_history(mock_retriever, mock_llm_cls):
    """With no prior history, FAISS is called with the original question directly."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.invoke.return_value = []
    mock_retriever.return_value = mock_retriever_instance

    mock_result = MagicMock()
    mock_result.content = "A bond is a fixed-income instrument."
    mock_llm_cls.return_value.invoke.return_value = mock_result

    from agents.rag_agent import answer
    answer("What is a bond?", history=[])

    # Only one invoke call — the final answer (no rewrite call)
    assert mock_llm_cls.return_value.invoke.call_count == 1
    mock_retriever_instance.invoke.assert_called_once_with("What is a bond?")


@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_rewrite_called_with_history(mock_retriever, mock_llm_cls):
    """With prior history, the LLM is called twice: once to rewrite, once to answer."""
    from langchain_core.messages import HumanMessage, AIMessage

    mock_retriever_instance = MagicMock()
    mock_retriever_instance.invoke.return_value = []
    mock_retriever.return_value = mock_retriever_instance

    rewrite_result = MagicMock()
    rewrite_result.content = "What are the risks of bonds?"
    answer_result = MagicMock()
    answer_result.content = "Bonds carry interest rate and credit risk."

    mock_llm_cls.return_value.invoke.side_effect = [rewrite_result, answer_result]

    history = [
        HumanMessage(content="What is a bond?"),
        AIMessage(content="A bond is a fixed-income instrument."),
        HumanMessage(content="Tell me more about that."),
    ]

    from agents.rag_agent import answer
    response = answer("Tell me more about that.", history=history)

    # LLM called twice: rewrite + final answer
    assert mock_llm_cls.return_value.invoke.call_count == 2
    # FAISS retrieval used the rewritten query, not the vague follow-up
    mock_retriever_instance.invoke.assert_called_once_with("What are the risks of bonds?")
    assert "risk" in response.lower()


@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_rewritten_query_differs_from_original(mock_retriever, mock_llm_cls):
    """The rewritten query passed to FAISS is the LLM's rewrite, not the raw follow-up."""
    from langchain_core.messages import HumanMessage, AIMessage

    mock_retriever_instance = MagicMock()
    mock_retriever_instance.invoke.return_value = []
    mock_retriever.return_value = mock_retriever_instance

    rewrite_result = MagicMock()
    rewrite_result.content = "compound interest calculation formula"
    answer_result = MagicMock()
    answer_result.content = "Compound interest grows exponentially over time."

    mock_llm_cls.return_value.invoke.side_effect = [rewrite_result, answer_result]

    history = [
        HumanMessage(content="Explain compound interest."),
        AIMessage(content="Compound interest is interest on interest."),
        HumanMessage(content="How is it calculated?"),
    ]

    from agents.rag_agent import answer
    answer("How is it calculated?", history=history)

    retrieval_query = mock_retriever_instance.invoke.call_args[0][0]
    assert retrieval_query == "compound interest calculation formula"
    assert retrieval_query != "How is it calculated?"
