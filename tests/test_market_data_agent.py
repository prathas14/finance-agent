import pytest
from unittest.mock import patch
from agents.market_data_agent import get_quote


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
