import pytest
from unittest.mock import patch, MagicMock
from agents.market_data_agent import get_quote, get_history, get_news


MOCK_QUOTE = {
    "symbol": "AAPL",
    "price": 195.50,
    "change": 1.20,
    "change_pct": "0.62%",
    "volume": 52000000,
    "latest_trading_day": "2026-05-26",
}


@patch("agents.market_data_agent.alpha_vantage.get_quote", return_value=MOCK_QUOTE)
def test_get_quote_from_alpha_vantage(mock_av):
    quote = get_quote("AAPL")
    assert quote["symbol"] == "AAPL"
    assert quote["price"] == 195.50
    mock_av.assert_called_once_with("AAPL")


@patch("agents.market_data_agent.alpha_vantage.get_quote", side_effect=RuntimeError("rate limit"))
@patch("agents.market_data_agent.yfinance_tool.get_quote_fallback", return_value={**MOCK_QUOTE, "source": "yfinance"})
def test_get_quote_falls_back_to_yfinance(mock_yf, mock_av):
    quote = get_quote("AAPL")
    assert quote["source"] == "yfinance"
    mock_yf.assert_called_once_with("AAPL")


@patch("agents.market_data_agent.yfinance_tool.get_history", return_value=[{"date": "2026-05-01", "close": 190.0}])
def test_get_history(mock_hist):
    result = get_history("AAPL", "1mo")
    assert len(result) == 1
    assert result[0]["close"] == 190.0
    mock_hist.assert_called_once_with("AAPL", "1mo")


@patch("agents.market_data_agent.alpha_vantage.get_news_sentiment", return_value=[{"title": "AAPL hits ATH"}])
def test_get_news_returns_feed(mock_news):
    result = get_news("AAPL")
    assert result[0]["title"] == "AAPL hits ATH"


@patch("agents.market_data_agent.alpha_vantage.get_news_sentiment", side_effect=RuntimeError("error"))
def test_get_news_returns_empty_on_error(mock_news):
    result = get_news("AAPL")
    assert result == []


@patch("agents.market_data_agent.ChatOpenAI")
def test_answer_no_ticker_returns_guidance(mock_llm_cls):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "NONE"
    mock_llm_cls.return_value = mock_llm

    from agents.market_data_agent import answer
    response = answer("tell me something about the market")
    assert "symbol" in response.lower() or "specify" in response.lower()


@patch("agents.market_data_agent.alpha_vantage.get_quote", side_effect=RuntimeError("fail"))
@patch("agents.market_data_agent.yfinance_tool.get_quote_fallback", side_effect=RuntimeError("fail"))
@patch("agents.market_data_agent.ChatOpenAI")
def test_answer_handles_ticker_error(mock_llm_cls, mock_yf, mock_av):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "AAPL"
    mock_llm_cls.return_value = mock_llm

    from agents.market_data_agent import answer
    response = answer("What is AAPL trading at?")
    assert "educational purposes" in response or "financial advice" in response