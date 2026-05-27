from langchain_google_genai import ChatGoogleGenerativeAI
from tools import alpha_vantage, yfinance_tool

DISCLAIMER = "\n\n⚠️ *This is for educational purposes only and not financial advice.*"


def get_quote(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    try:
        return alpha_vantage.get_quote(symbol)
    except Exception:
        return yfinance_tool.get_quote_fallback(symbol)


def get_history(symbol: str, period: str = "1mo") -> list[dict]:
    return yfinance_tool.get_history(symbol, period)


def get_news(symbol: str) -> list[dict]:
    try:
        return alpha_vantage.get_news_sentiment(symbol)
    except Exception:
        return []


def answer(query: str) -> str:
    """Interpret a natural language market query and respond."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

    extract_prompt = f"""Extract the stock ticker symbol(s) from this query.
Return ONLY the ticker symbol(s) comma-separated, nothing else.
If no ticker is mentioned, return NONE.

Query: {query}
Tickers:"""

    tickers_raw = llm.invoke(extract_prompt).content.strip()

    if tickers_raw == "NONE" or not tickers_raw:
        return "Please specify a stock symbol (e.g. 'What is the price of AAPL?')" + DISCLAIMER

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    results = []

    for ticker in tickers[:3]:
        try:
            quote = get_quote(ticker)
            news = get_news(ticker)
            news_headlines = [n.get("title", "") for n in news[:3]]

            results.append({
                "symbol": ticker,
                "quote": quote,
                "recent_news": news_headlines,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    context = str(results)
    summary_prompt = f"""You are a market data assistant. Based on the following live data, answer the user's query.
Be factual and concise. Include price, change, and relevant news if available.

Live Data: {context}
User Query: {query}

Answer:"""

    response = llm.invoke(summary_prompt)
    return response.content + DISCLAIMER
