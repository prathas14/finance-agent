import os
import requests
from cachetools import TTLCache, cached
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

_cache = TTLCache(maxsize=128, ttl=1800)  # 30-min TTL


def _get(params: dict) -> dict:
    key = str(sorted(params.items()))
    if key in _cache:
        return _cache[key]
    try:
        resp = requests.get(BASE_URL, params={**params, "apikey": API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "Note" in data or "Information" in data:
            raise RuntimeError("Alpha Vantage rate limit hit")
        _cache[key] = data
        return data
    except Exception as e:
        raise RuntimeError(f"Alpha Vantage error: {e}")


def get_quote(symbol: str) -> dict:
    data = _get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    q = data.get("Global Quote", {})
    if not q:
        raise RuntimeError(f"No data for {symbol}")
    return {
        "symbol": q.get("01. symbol"),
        "price": float(q.get("05. price", 0)),
        "change": float(q.get("09. change", 0)),
        "change_pct": q.get("10. change percent", "0%"),
        "volume": int(q.get("06. volume", 0)),
        "latest_trading_day": q.get("07. latest trading day"),
    }


def get_company_overview(symbol: str) -> dict:
    return _get({"function": "OVERVIEW", "symbol": symbol})


def get_news_sentiment(symbol: str) -> list[dict]:
    data = _get({"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": "5"})
    return data.get("feed", [])


def get_time_series_daily(symbol: str, outputsize: str = "compact") -> dict:
    return _get({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize})
