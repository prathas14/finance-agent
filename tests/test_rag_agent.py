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
