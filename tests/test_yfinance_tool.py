import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


def _mock_ticker(price=150.0, volume=1000000, history_rows=None):
    ticker = MagicMock()
    ticker.fast_info.last_price = price
    ticker.fast_info.last_volume = volume
    if history_rows is None:
        history_rows = [{"Close": 150.0}]
    index = pd.to_datetime(["2026-05-01"] * len(history_rows))
    ticker.history.return_value = pd.DataFrame(history_rows, index=index)
    ticker.info = {"longName": "Apple Inc.", "sector": "Technology"}
    return ticker


@patch("tools.yfinance_tool.yf.Ticker")
def test_get_quote_fallback(mock_ticker_cls):
    mock_ticker_cls.return_value = _mock_ticker(price=195.5, volume=52000000)
    from tools.yfinance_tool import get_quote_fallback
    result = get_quote_fallback("AAPL")
    assert result["symbol"] == "AAPL"
    assert result["price"] == 195.5
    assert result["volume"] == 52000000
    assert result["source"] == "yfinance"


@patch("tools.yfinance_tool.yf.Ticker")
def test_get_history(mock_ticker_cls):
    mock_ticker_cls.return_value = _mock_ticker(history_rows=[{"Close": 190.0}, {"Close": 192.5}])
    from tools.yfinance_tool import get_history
    result = get_history("AAPL", "1mo")
    assert len(result) == 2
    assert result[0]["close"] == 190.0
    assert result[1]["close"] == 192.5
    assert "date" in result[0]


@patch("tools.yfinance_tool.yf.Ticker")
def test_get_company_info(mock_ticker_cls):
    mock_ticker_cls.return_value = _mock_ticker()
    from tools.yfinance_tool import get_company_info
    info = get_company_info("AAPL")
    assert info["longName"] == "Apple Inc."
    assert info["sector"] == "Technology"
