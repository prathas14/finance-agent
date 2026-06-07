import pytest
from unittest.mock import patch, MagicMock
from cachetools import TTLCache


def _make_response(json_data, status=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@patch("tools.alpha_vantage.requests.get")
def test_get_quote_success(mock_get):
    mock_get.return_value = _make_response({
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "195.50",
            "09. change": "1.20",
            "10. change percent": "0.62%",
            "06. volume": "52000000",
            "07. latest trading day": "2026-05-26",
        }
    })
    import tools.alpha_vantage as av
    av._cache.clear()
    result = av.get_quote("AAPL")
    assert result["symbol"] == "AAPL"
    assert result["price"] == 195.50
    assert result["volume"] == 52000000


@patch("tools.alpha_vantage.requests.get")
def test_get_quote_empty_raises(mock_get):
    mock_get.return_value = _make_response({"Global Quote": {}})
    import tools.alpha_vantage as av
    av._cache.clear()
    with pytest.raises(RuntimeError, match="No data"):
        av.get_quote("FAKE")


@patch("tools.alpha_vantage.requests.get")
def test_rate_limit_note_raises(mock_get):
    mock_get.return_value = _make_response({"Note": "Thank you for using Alpha Vantage!"})
    import tools.alpha_vantage as av
    av._cache.clear()
    with pytest.raises(RuntimeError, match="rate limit"):
        av.get_quote("AAPL")


@patch("tools.alpha_vantage.requests.get")
def test_rate_limit_information_raises(mock_get):
    mock_get.return_value = _make_response({"Information": "API limit reached."})
    import tools.alpha_vantage as av
    av._cache.clear()
    with pytest.raises(RuntimeError, match="rate limit"):
        av.get_quote("AAPL")


@patch("tools.alpha_vantage.requests.get")
def test_cache_hit_skips_request(mock_get):
    import tools.alpha_vantage as av
    av._cache.clear()
    data = {"Global Quote": {
        "01. symbol": "MSFT", "05. price": "420.00", "09. change": "0.50",
        "10. change percent": "0.12%", "06. volume": "30000000",
        "07. latest trading day": "2026-05-26",
    }}
    mock_get.return_value = _make_response(data)
    av.get_quote("MSFT")
    av.get_quote("MSFT")
    assert mock_get.call_count == 1


@patch("tools.alpha_vantage.requests.get")
def test_get_news_sentiment(mock_get):
    mock_get.return_value = _make_response({"feed": [{"title": "AAPL surges"}]})
    import tools.alpha_vantage as av
    av._cache.clear()
    result = av.get_news_sentiment("AAPL")
    assert result[0]["title"] == "AAPL surges"


@patch("tools.alpha_vantage.requests.get")
def test_get_company_overview(mock_get):
    mock_get.return_value = _make_response({"Symbol": "AAPL", "Sector": "Technology"})
    import tools.alpha_vantage as av
    av._cache.clear()
    result = av.get_company_overview("AAPL")
    assert result["Sector"] == "Technology"


@patch("tools.alpha_vantage.requests.get")
def test_get_time_series_daily(mock_get):
    mock_get.return_value = _make_response({"Time Series (Daily)": {"2026-05-26": {"4. close": "195.50"}}})
    import tools.alpha_vantage as av
    av._cache.clear()
    result = av.get_time_series_daily("AAPL")
    assert "Time Series (Daily)" in result