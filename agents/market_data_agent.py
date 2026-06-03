from langchain_openai import ChatOpenAI
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


def answer(query: str, history: list = []) -> str:
    """Interpret a natural language market query and respond."""
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Use history to resolve follow-up ticker references (e.g. "what about its news?")
    history_text = ""
    if history:
        prior = [m for m in history[:-1] if hasattr(m, "content")]
        if prior:
            history_text = "\n".join(
                f"{'User' if m.__class__.__name__ == 'HumanMessage' else 'Assistant'}: {m.content}"
                for m in prior[-6:]  # last 3 turns
            )

    extract_prompt = f"""Extract the stock ticker symbol(s) from this query.
Return ONLY the ticker symbol(s) comma-separated, nothing else.
If no ticker is mentioned but one was discussed recently in the conversation, use that.
If still unclear, return NONE.
{f'Recent conversation:{chr(10)}{history_text}' if history_text else ''}
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
            results.append({"symbol": ticker, "quote": quote, "recent_news": news_headlines})
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    prior = history[:-1] if history else []
    system = SystemMessage(content=(
        "You are a market data assistant. Based on the following live data, answer the user's query. "
        "Be factual and concise. Include price, change, and relevant news if available.\n\n"
        f"Live Data: {results}"
    ))
    messages = [system] + prior + [HumanMessage(content=query)]

    response = llm.invoke(messages)
    return response.content + DISCLAIMER
