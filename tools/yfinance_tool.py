import yfinance as yf


def get_quote_fallback(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    return {
        "symbol": symbol,
        "price": getattr(info, "last_price", None),
        "change": None,
        "change_pct": None,
        "volume": getattr(info, "last_volume", None),
        "latest_trading_day": None,
        "source": "yfinance",
    }


def get_history(symbol: str, period: str = "1mo") -> list[dict]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    return [
        {"date": str(date.date()), "close": round(row["Close"], 2)}
        for date, row in hist.iterrows()
    ]


def get_company_info(symbol: str) -> dict:
    return yf.Ticker(symbol).info
