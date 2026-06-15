"""
MCP Server for AI Finance Assistant.

Exposes four tools to Claude Desktop via the Model Context Protocol (stdio transport):
  - ask_finance      : general query, auto-routed by the orchestrator
  - get_market_data  : live stock quote + news for a ticker
  - plan_goal        : savings goal / compound-interest calculation
  - analyze_portfolio: summarise a portfolio (uses built-in sample if none supplied)

Run directly:
    python mcp_server.py

Claude Desktop config  (~/.../Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "finance-agent": {
          "command": "/absolute/path/to/venv/bin/python",
          "args": ["/absolute/path/to/finance-agent/mcp_server.py"]
        }
      }
    }
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path so agent imports resolve correctly
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("finance-agent")


# ---------------------------------------------------------------------------
# Tool 1 — General query (orchestrator auto-routes to the right agent)
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_finance(query: str, session_id: str = "mcp-default") -> str:
    """
    Answer any financial question by routing it to the correct specialist agent.

    The orchestrator classifies the query as one of:
    - rag       : financial education and definitions
    - portfolio : portfolio analysis and net-worth
    - market    : live stock prices and news
    - goal      : savings goals and compound-interest projections

    Args:
        query:      The user's question in plain English.
        session_id: Optional identifier to maintain multi-turn memory across calls.

    Returns:
        A plain-text answer from the appropriate agent.
    """
    from agents.orchestrator import chat
    return chat(query, session_id=session_id)


# ---------------------------------------------------------------------------
# Tool 2 — Market data (live quote + news for up to 3 tickers)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_market_data(query: str) -> str:
    """
    Fetch live stock price, change, and recent news headlines for ticker(s).

    Accepts natural-language queries such as:
    - "What is the price of AAPL?"
    - "Compare TSLA and NVDA"
    - "Show me the latest news for MSFT"

    Args:
        query: Natural-language market query containing ticker symbol(s).

    Returns:
        A plain-text summary with price, % change, and news headlines.
    """
    from agents.market_data_agent import answer
    return answer(query)


# ---------------------------------------------------------------------------
# Tool 3 — Goal planning (savings / compound-interest)
# ---------------------------------------------------------------------------

@mcp.tool()
def plan_goal(
    query: str,
    goals: str = "[]",
) -> str:
    """
    Calculate savings timelines and compound-interest projections.

    Example queries:
    - "How long to save $50,000 if I contribute $500/month at 7% return?"
    - "What will $10,000 grow to in 20 years at 8%?"

    Args:
        query: Natural-language goal-planning question.
        goals: JSON array of goal objects, each with keys:
               name, target, current, monthly_contribution.
               Pass "[]" (default) to rely solely on the query text.

    Returns:
        A plain-text answer with specific numbers and a timeline.
    """
    from agents.goal_planning_agent import answer
    try:
        goals_list = json.loads(goals) if goals else []
    except json.JSONDecodeError:
        goals_list = []
    return answer(query, goals=goals_list)


# ---------------------------------------------------------------------------
# Tool 4 — Portfolio analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_portfolio(portfolio: str = "{}") -> str:
    """
    Summarise a portfolio: positions, allocation, gain/loss, and net worth.

    If no portfolio is supplied the server uses the built-in sample portfolio
    (knowledge_base/sample_portfolio.json) so you can test immediately.

    Args:
        portfolio: JSON object with keys:
                   holdings  — list of {symbol, shares, avg_cost, asset_class}
                   cash      — float
                   liabilities — list of {name, balance}
                   Pass "{}" (default) to use the sample portfolio.

    Returns:
        A plain-text 3-5 sentence portfolio summary with one actionable insight.
    """
    from agents.portfolio_agent import summarize
    try:
        portfolio_dict = json.loads(portfolio) if portfolio and portfolio != "{}" else None
    except json.JSONDecodeError:
        portfolio_dict = None
    return summarize(portfolio_dict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")