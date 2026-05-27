import pytest
from unittest.mock import patch, MagicMock


@patch("agents.rag_agent._get_chain")
def test_answer_returns_disclaimer(mock_chain):
    mock_result = MagicMock()
    mock_result.content = "An ETF is an exchange-traded fund."
    mock_chain.return_value.invoke.return_value = mock_result

    from agents.rag_agent import answer
    response = answer("What is an ETF?")

    assert "ETF" in response
    assert "educational purposes" in response or "financial advice" in response


@patch("agents.rag_agent._get_chain")
def test_answer_handles_exception(mock_chain):
    mock_chain.side_effect = Exception("Index not found")

    from agents.rag_agent import answer
    response = answer("What is a bond?")

    assert "error" in response.lower() or "RAG" in response
