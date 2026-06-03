import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from tools import alpha_vantage, yfinance_tool

DISCLAIMER = "\n\n⚠️ *This is for educational purposes only and not financial advice.*"
SAMPLE_PORTFOLIO_PATH = Path(__file__).parent.parent / "knowledge_base" / "sample_portfolio.json"


def _load_portfolio(portfolio: dict | None) -> dict:
    if portfolio:
        return portfolio
    return json.loads(SAMPLE_PORTFOLIO_PATH.read_text())


def _fetch_prices(holdings: list[dict]) -> dict[str, float]:
    prices = {}
    for h in holdings:
        symbol = h["symbol"]
        try:
            prices[symbol] = alpha_vantage.get_quote(symbol)["price"]
        except Exception:
            try:
                prices[symbol] = yfinance_tool.get_quote_fallback(symbol)["price"] or h["avg_cost"]
            except Exception:
                prices[symbol] = h["avg_cost"]
    return prices


def analyze(portfolio: dict | None = None) -> dict:
    p = _load_portfolio(portfolio)
    holdings = p["holdings"]
    prices = _fetch_prices(holdings)

    positions = []
    total_value = p.get("cash", 0)
    total_cost = 0

    for h in holdings:
        symbol = h["symbol"]
        price = prices[symbol]
        cost_basis = h["avg_cost"] * h["shares"]
        market_value = price * h["shares"]
        gain_loss = market_value - cost_basis
        gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis else 0

        positions.append({
            "symbol": symbol,
            "shares": h["shares"],
            "avg_cost": h["avg_cost"],
            "current_price": round(price, 2),
            "market_value": round(market_value, 2),
            "gain_loss": round(gain_loss, 2),
            "gain_loss_pct": round(gain_loss_pct, 2),
            "asset_class": h.get("asset_class", "unknown"),
        })
        total_value += market_value
        total_cost += cost_basis

    total_liabilities = sum(l["balance"] for l in p.get("liabilities", []))
    net_worth = total_value - total_liabilities

    allocation = {}
    for pos in positions:
        ac = pos["asset_class"]
        allocation[ac] = allocation.get(ac, 0) + pos["market_value"]
    allocation["cash"] = p.get("cash", 0)
    allocation_pct = {k: round(v / total_value * 100, 1) for k, v in allocation.items() if total_value > 0}

    total_gain_loss = total_value - total_cost - p.get("cash", 0)

    return {
        "positions": positions,
        "total_portfolio_value": round(total_value, 2),
        "total_cost_basis": round(total_cost, 2),
        "total_gain_loss": round(total_gain_loss, 2),
        "total_liabilities": round(total_liabilities, 2),
        "net_worth": round(net_worth, 2),
        "allocation": allocation,
        "allocation_pct": allocation_pct,
        "cash": p.get("cash", 0),
        "goals": p.get("goals", []),
    }


def summarize(portfolio: dict | None = None) -> str:
    try:
        data = analyze(portfolio)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        prompt = f"""You are a portfolio analyst. Summarize the following portfolio data in plain English.
Highlight key strengths, risks, and one actionable suggestion. Be concise (3-5 sentences).

Portfolio Data:
{json.dumps(data, indent=2)}

Summary:"""

        result = llm.invoke(prompt)
        return result.content + DISCLAIMER
    except Exception as e:
        return f"Portfolio agent error: {e}{DISCLAIMER}"
